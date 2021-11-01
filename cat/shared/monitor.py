# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Monitor figure plotting module.

Usage:
    in working directory:
    python3 cat/shared/monitor.py <path to my checkpoint>
"""

import time
import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Any, Dict, List

import torch

FILE_WRITER = r"training.summary"


class BaseSummary():
    def __init__(self, src: dict = None) -> None:
        if src is None:
            self._values = []
            self._time = []
            self._cnt = 0
        else:
            self.load(src)

    def dump(self) -> dict:
        return {'val': self._values, 'time': self._time, 'cnt': self._cnt}

    def load(self, src: dict):
        self._values = src['val']
        self._time = src['time']
        self._cnt = src['cnt']

    def update(self, value: Any):
        self._values.append(value)
        self._time.append(time.time())
        self._cnt += 1

    def merge(self, appd_summary):
        if self._cnt > 0 and appd_summary._cnt > 0:
            if self._time[-1] > appd_summary._time[0]:
                raise RuntimeError(
                    f"Trying to merge conflict Summary: s0_end > s1_beg: {self._time[-1]} > {appd_summary._time[0]}")

        self._time += appd_summary._time
        self._values += appd_summary._values
        self._cnt += appd_summary._cnt


class MonitorWriter():
    def __init__(self, path: str = './') -> None:
        if os.path.isdir(path):
            path = os.path.join(path, FILE_WRITER)
        else:
            # assume path is file-like
            pass

        self._default_path = path
        self.summaries = {}   # type: Dict[str, BaseSummary]

    def __getitem__(self, index):
        return self.summaries[index]

    def addWriter(self, names: Union[List[str], str]):
        if isinstance(names, str):
            names = [names]

        for m in names:
            assert isinstance(
                m, str), f"expect metric type to be str, instead of {type(m)}"
            if m in self.summaries:
                continue
            self.summaries[m] = BaseSummary()

    def update(self, name: Union[str, dict], value: Any = None):
        """update summary writer

        Example:
            >>> writer.update('loss', 0.1)  # one per invoking
            >>> writer.update({'loss': 0.1, 'acc': 0.5})    # update a batch of values
        """
        if value is None:
            assert isinstance(
                name, dict), f"update a batch of values only accept dict as argument, instead {name}"
            toupdate = name
        else:
            toupdate = {name: value}

        for metric, val in toupdate.items():
            assert isinstance(
                metric, str), f"expect metric type to be str, instead of {type(metric)}"
            assert metric in self.summaries, f"try to update {metric}, but expected one of {list(self.summaries.keys())}"
            self.summaries[metric].update(val)

    def empty(self):
        metrics = list(self.summaries.keys())
        del self.summaries
        self.summaries = {}
        for m in metrics:
            self.addWriter(m)

    def export(self, path: str = None):
        """export writer, if path is None, export to self._default_path"""

        if path is None:
            path = self._default_path
        elif os.path.isdir(path):
            path = os.path.join(path, FILE_WRITER)

        if os.path.isfile(path):
            prev_writer = MonitorWriter()
            prev_writer.load(path)
            prev_writer.merge(self)
            writer = prev_writer
        else:
            writer = self
            prev_writer = None

        with open(path, 'wb') as fo:
            pickle.dump({
                'path': writer._default_path,
                'summaries': {metric: smr.dump() for metric, smr in writer.summaries.items()}
            }, fo)

        del prev_writer
        self.empty()

    def load(self, path: str = None):
        """load writer, if path is None, load from self._default_path"""

        if path is None:
            path = self._default_path
        elif os.path.isdir(path):
            path = os.path.join(path, FILE_WRITER)

        assert os.path.isfile(
            path), f"{self.__class__.__name__}: trying to load from invalid file {path}"

        with open(path, 'rb') as fi:
            check = pickle.load(fi)

            self._default_path = check['path']
            self.summaries = {}
            for m, smr in check['summaries'].items():
                self.summaries[m] = BaseSummary(smr)

    def merge(self, appd_writer):

        assert list(self.summaries.keys()) == list(appd_writer.summaries.keys(
        )), f"{self.__class__.__name__}: merge failed due to mismatch keys {list(self.summaries.keys())} {list(appd_writer.summaries.keys())}"

        for m in self.summaries:
            self.summaries[m].merge(appd_writer[m])

    def visualize(self, fig_path: str = None) -> str:
        self.export()
        return plot_monitor(self._default_path, o_path=fig_path, pt_like=False)

# FIXME : deprecate this


def conver2new(path_old: str, path_new: str):
    import time

    assert os.path.isfile(path_old), path_old
    assert os.path.isdir(path_new), path_new
    assert not os.path.isfile(os.path.join(
        path_new, FILE_WRITER)), f"File {path_new}/{FILE_WRITER} exits."

    prev_check = torch.load(path_old, map_location='cpu')['log']
    writer = MonitorWriter(path_new)

    writer.addWriter(['train:loss', 'train:lr', 'eval:loss'])
    T = [0]
    for hist in prev_check['log_train'][1:]:
        _, _, loss, lr, t = hist
        writer.update({
            'train:loss': loss,
            'train:lr': lr
        })
        T.append(t + T[-1])

    reset_m = time.time()
    T = [t + reset_m - T[-1] for t in T]
    for metric in ['train:loss', 'train:lr']:
        writer[metric]._time = T[1:]

    for hist in prev_check['log_eval'][1:]:
        loss, _ = hist
        writer.update('eval:loss', loss)

    writer.export()
    return writer._default_path


def read_from_check(path: str, pt_like: bool = False) -> Tuple[np.array, np.array, int, int]:
    if pt_like:
        # FIXME (huahuan): deprecated in the next release
        check = torch.load(path, map_location='cpu')['log']

        '''
        check: OrderedDict({
                'log_train': ['epoch,loss,loss_real,net_lr,time'],
                'log_eval': ['loss_real,time']
            })
        '''

        df_train = np.array(check['log_train'][1:])
        df_train = {
            'loss': df_train[:, 2],
            'lr': df_train[:, 3],
            'time': df_train[:, 4],
        }
        df_eval = np.array(check['log_eval'][1:])
        df_eval = {
            'loss': df_eval[:, 0],
            'time': df_eval[:, 1]
        }
        num_batches = df_train['loss'].shape[0]
        num_epochs = df_eval['loss'].shape[0]
        return df_train, df_eval, num_batches, num_epochs
    else:
        tmp_monitor = MonitorWriter()
        tmp_monitor.load(path)
        tr_m = {
            'loss': np.asarray(tmp_monitor['train:loss']._values),
            'lr': np.asarray(tmp_monitor['train:lr']._values),
            'time': np.asarray(tmp_monitor['train:loss']._time)
        }
        eval_m = {
            'loss': np.asarray(tmp_monitor['eval:loss']._values),
            'time': None
        }
        # FIXME : for compatible of old API
        tr_m['time'][1:] = tr_m['time'][1:] - tr_m['time'][:-1]
        tr_m['time'][0] = 0.0
        eval_m['time'] = np.zeros_like(eval_m['loss'])
        return tr_m, eval_m, tmp_monitor['train:loss']._cnt, tmp_monitor['eval:loss']._cnt


def draw_time(ax: plt.Axes, scalars: Union[np.array, list], num_steps: int, num_epochs: int, eval_time: Union[np.array, list], prop_box=True):
    batch_per_epoch = num_steps//num_epochs
    accum_time = scalars[:]
    for i in range(1, len(accum_time)):
        accum_time[i] += accum_time[i-1]
        if (i + 1) % batch_per_epoch == 0:
            accum_time[i] += eval_time[(i+1)//batch_per_epoch-1]
    del batch_per_epoch
    accum_time = [x/3600 for x in accum_time]
    ax.plot(accum_time)

    if prop_box:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        speed = accum_time[-1]/num_epochs
        if speed < 1.:
            speed = speed * 60
            if speed < 1.:
                speed = speed * 60
                timestr = "{:.0f}sec/epoch".format(speed)
            else:
                timestr = "{:.1f}min/epoch".format(speed)
        else:
            timestr = "{:.2f}h/epoch".format(speed)
        ax.text(0.05, 0.95, timestr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', bbox=props)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('Total time / h')
    return ax


def draw_tr_loss(ax: plt.Axes, scalars: Union[np.array, list], smooth_value: float = 0.9):
    assert smooth_value >= 0. and smooth_value < 1.
    running_mean = [scalars[0]]
    res_smooth = 1 - smooth_value
    for i in range(1, len(scalars)):
        running_mean.append(
            running_mean[i-1]*smooth_value+res_smooth*scalars[i])

    min_loss = min(running_mean)
    if min_loss <= 0.:
        ax.plot(running_mean)
    else:
        ax.semilogy(running_mean)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(True, ls='--')
    ax.set_ylabel('Training loss')
    ax.set_xlabel("Step")
    return ax


def draw_dev_loss(ax: plt.Axes, scalars: Union[np.array, list], num_epochs: int, prop_box=True):

    min_loss = min(scalars)
    if min_loss <= 0.:
        ax.plot([i+1 for i in range(num_epochs)], scalars)
    else:
        ax.semilogy([i+1 for i in range(num_epochs)], scalars)

    # ax.axhline(y=min_loss, ls='--', color='black', alpha=0.5)
    if prop_box:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        textstr = '\n'.join([
            "min={:.2f}".format(min_loss),
            f"{num_epochs} epoch"
        ])
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.grid(True, ls='--')
    ax.set_ylabel('Dev loss')
    ax.set_xlabel('Epoch')
    return ax


def draw_lr(ax: plt.Axes, scalars: Union[np.array, list]):

    ax.semilogy(scalars)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('learning rate')
    return ax


def plot_monitor(path: str, title: str = None, interactive_show=False, o_path: str = None, pt_like: bool = True) -> str:
    """Plot the monitor log files

    Args:
        path (str): directory of log files
        title (str, optional): title name (title of ploting)
        interactive_show (bool, optional): specify whether plot in interactive mode. Default False. 
    """

    if title is None:
        title = ' '

    df_train, df_eval, num_batches, num_epochs = read_from_check(path, pt_like)

    _, axes = plt.subplots(2, 2)

    # Time
    draw_time(axes[0][0], df_train['time'],
              num_batches, num_epochs, df_eval['time'])

    # Learning rate
    draw_lr(axes[0][1], df_train['lr'])

    # Training loss and moving average
    draw_tr_loss(axes[1][0], df_train['loss'])

    # Dev loss
    draw_dev_loss(axes[1][1], df_eval['loss'], num_epochs)

    # Global settings

    plt.suptitle(title)
    plt.tight_layout()
    if interactive_show:
        outpath = None
        plt.show()
    else:
        if o_path is None:
            direc = os.path.dirname(path)
            outpath = os.path.join(direc, 'monitor.png')
        else:
            if os.path.isdir(o_path):
                outpath = os.path.join(o_path, 'monitor.png')
            else:
                assert os.path.isdir(os.path.dirname(o_path))
                outpath = o_path
        plt.savefig(outpath, dpi=250)
    plt.close()
    return outpath


def cmp(check0: str, check1: str, legends: Union[Tuple[str, str], None] = None, title: str = ' ', o_path=None, pt_like: bool = True):
    assert os.path.isfile(check0), f"{check0} is not a file."
    assert os.path.isfile(check1), f"{check1} is not a file."

    _, axes = plt.subplots(2, 2)

    if legends is None:
        legends = ['1', '2']

    for clog in [check0, check1]:

        df_train, df_eval, num_batches, num_epochs = read_from_check(
            clog, pt_like)

        # Time
        draw_time(axes[0][0], df_train['time'],
                  num_batches, num_epochs, df_eval['time'], prop_box=False)

        # Learning rate
        draw_lr(axes[0][1], df_train['lr'])

        # Training loss and moving average
        draw_tr_loss(axes[1][0], df_train['loss'])

        # Dev loss
        draw_dev_loss(axes[1][1], df_eval['loss'], num_epochs, prop_box=False)

    axes[0][1].legend(legends, fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()

    legends = [x.replace(' ', '_') for x in legends]
    if o_path is None:
        outpath = os.path.join('.', 'compare-{}-{}.png'.format(*legends))
    else:
        if os.path.isdir(o_path):
            outpath = os.path.join(
                o_path, 'compare-{}-{}.png'.format(*legends))
        else:
            assert os.path.isdir(os.path.dirname(o_path))
            outpath = o_path
    plt.savefig(outpath, dpi=250)
    plt.close()
    return outpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str, help="Location of log files.")
    parser.add_argument("--title", type=str, default=None,
                        help="Configure the plotting title.")
    parser.add_argument("--cmp", type=str, default=None,
                        help="Same format as log, compared one.")
    parser.add_argument("--cmplegend", type=str, default='1-2',
                        help="Legend for two comparing figures, split by '-'. Default: 1-2")
    parser.add_argument("-o", type=str, default=None, dest="o_path",
                        help="Output figure path.")
    parser.add_argument("--convert", action="store_true",
                        help="Convert old monitor to new one.")
    args = parser.parse_args()

    isold = (args.log[-3:] == '.pt')

    if args.convert:
        if not isold:
            raise RuntimeError

        print("> Ouput file: {}".format(conver2new(args.log, args.o_path)))
        exit(0)

    if args.cmp is None:
        plot_monitor(args.log, title=args.title,
                     o_path=args.o_path, pt_like=isold)
    else:
        legends = args.cmplegend.split('-')
        assert len(legends) == 2
        cmp(args.log, args.cmp, legends=legends,
            title=args.title, o_path=args.o_path, pt_like=isold)
