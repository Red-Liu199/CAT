# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Monitor figure plotting module.

Usage:
    in working directory:
    python3 cat/shared/monitor.py <path to my logfile>
"""

import time
import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Any, Dict, List

FILE_WRITER = r"training.summary"
BASE_METRIC = ['train:loss', 'train:lr', 'eval:loss']


class BaseSummary():
    def __init__(self, src: dict = None) -> None:
        if src is None:
            self._values = []   # type: List[Any]
            self._time = []     # type: List[float]
            self._cnt = 0
        else:
            self.load(src)

    @property
    def data(self) -> dict:
        return self.dump()

    def dump(self) -> dict:
        return {
            'val': self._values,
            'time': self._time,
            'cnt': self._cnt
        }

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
    '''Monitor writer

    Args:
        path (str): directory or path to resuming file.
    '''

    def __init__(self, path: str = './') -> None:
        if os.path.isfile(path):
            # assume path is file-like
            self.load(path)
            return
        elif os.path.isdir(path):
            path = os.path.join(path, FILE_WRITER)

        self._default_path = path
        self.summaries = {}   # type: Dict[str, BaseSummary]

    def __contains__(self, index):
        return index in self.summaries

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
            # update a batch of values
            >>> writer.update({'loss': 0.1, 'acc': 0.5})
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
            prev_writer = MonitorWriter(path)
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
        return plot_monitor(self._default_path, o_path=fig_path)


def draw_time(ax: plt.Axes, summary: BaseSummary, n_epoch: int = -1, prop_box=True):
    d_time = np.asarray(summary.data['time'])
    d_time -= d_time[0]
    d_time /= 3600.0

    if n_epoch == -1:
        ax.plot(d_time)
        fmt = 'iter'
        ylabel = 'Total time (h)'
        ax.ticklabel_format(axis="x", style="sci",
                            scilimits=(0, 0), useMathText=True)
        speed = d_time[-1]/d_time.shape[0]
    else:
        epoch_time = []
        step_per_epoch = d_time.shape[0]//n_epoch
        for i in range(n_epoch):
            epoch_time.append(
                d_time[(i+1)*step_per_epoch-1] - d_time[i*step_per_epoch])

        ax.plot([1+x for x in range(n_epoch)], epoch_time, '.', markersize=2)
        # plot a dummy marker to set ylim to 0.0
        ax.scatter(1, 0.5*min(epoch_time), alpha=0.0)
        fmt = 'epoch'
        ylabel = 'Time (hour)'
        speed = sum(epoch_time)/n_epoch

    if prop_box:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        if speed < 1.:
            speed = speed * 60
            if speed < 1.:
                speed = speed * 60
                timestr = "{:.0f} sec/{}".format(speed, fmt)
            else:
                timestr = "{:.1f} min/{}".format(speed, fmt)
        else:
            timestr = "{:.2f} h/{}".format(speed, fmt)
        ax.text(0.95, 0.05, timestr, transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.grid(ls='--')
    ax.set_ylabel(ylabel)
    return ax


def draw_tr_loss(ax: plt.Axes, summary: BaseSummary, smooth_value: float = 0.9):
    scalars = np.asarray(summary.data['val'])

    assert smooth_value >= 0. and smooth_value < 1.
    res_smooth = 1 - smooth_value

    running_mean = np.zeros_like(scalars)
    running_mean[0] = scalars[0]
    for i in range(1, len(scalars)):
        running_mean[i] = smooth_value * \
            running_mean[i-1] + res_smooth * scalars[i]

    min_loss = min(running_mean)
    if min_loss <= 0. or (max(scalars) / min(scalars) < 10.):
        ax.plot(scalars, color='C0', alpha=0.25)
        ax.plot(running_mean, color='C0')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax.semilogy(scalars, color='C0', alpha=0.25)
        ax.semilogy(running_mean, color='C0')

    ax.ticklabel_format(axis="x", style="sci",
                        scilimits=(0, 0), useMathText=True)
    ax.grid(True, ls='--', which='both')
    ax.set_ylabel('Training loss')
    ax.set_xlabel("Step")
    return ax


def draw_dev_loss(ax: plt.Axes, summary: BaseSummary, prop_box=True):

    scalars = np.asarray(summary.data['val'])
    num_epochs = summary.data['cnt']
    min_loss = min(scalars)
    if min_loss <= 0. or (max(scalars)/min_loss < 10.):
        ax.plot([i+1 for i in range(num_epochs)], scalars)
    else:
        ax.semilogy([i+1 for i in range(num_epochs)], scalars)

    if prop_box:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        textstr = '\n'.join([
            "min={:.2f}".format(min_loss),
            f"{num_epochs} epoch"
        ])
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.grid(True, ls='--', which='both')
    ax.set_ylabel('Dev loss')
    ax.set_xlabel('Epoch')
    return ax


def draw_lr(ax: plt.Axes, summary: BaseSummary):

    ax.plot(summary.data['val'])

    ax.ticklabel_format(axis="x", style="sci",
                        scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="y", style="sci",
                        scilimits=(0, 0), useMathText=True)
    ax.grid(ls='--', which='both')
    ax.set_ylabel('learning rate')
    return ax


def draw_any(ax: plt.Axes, summary: BaseSummary, _name: str = ''):
    ax.plot(summary.data['val'])
    ax.ticklabel_format(axis="x", style="sci",
                        scilimits=(0, 0), useMathText=True)
    ax.grid(ls='--')
    ax.set_ylabel(_name)
    return ax


def plot_monitor(path: str, o_path: str = None, title: str = None, interactive_show=False) -> str:
    """Plot the monitor log files

    Args:
        path (str): directory of log files
        title (str, optional): title name (title of ploting)
        interactive_show (bool, optional): specify whether plot in interactive mode. Default False.
    """

    if title is None:
        title = ' '

    log_writer = MonitorWriter(path)
    apd = []
    for k in log_writer.summaries:
        if k not in BASE_METRIC:
            apd.append(k)

    n_row = 2 + (len(apd) // 2 + len(apd) % 2)
    n_col = 2

    _, axes = plt.subplots(n_row, n_col, figsize=(
        3*n_col, 2.2*n_row), constrained_layout=True)

    # Learning rate
    draw_lr(axes[0][0], log_writer['train:lr'])
    axes[0][0].set_xticklabels([])

    # Time
    draw_time(axes[0][1], log_writer['train:loss'],
              log_writer['eval:loss'].data['cnt'], True)
    axes[0][1].set_xticklabels([])

    # Training loss and moving average
    draw_tr_loss(axes[1][0], log_writer['train:loss'])

    # Dev loss
    draw_dev_loss(axes[1][1], log_writer['eval:loss'])

    # custom metric
    for i, k in enumerate(apd):
        r, c = (4+i)//2, (4+i) % 2
        draw_any(axes[r][c], log_writer[k], k)
    # rm the empty subplot
    if len(apd) % 2 != 0:
        axes[-1][-1].set_axis_off()

    # Global settings
    plt.suptitle(title)
    # plt.tight_layout()
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
        plt.savefig(outpath, dpi=200, facecolor="w")
    plt.close()
    return outpath


def cmp(checks: List[str], legends: Union[List[str], None] = None, title: str = ' ', o_path=None):

    for c in checks:
        assert os.path.isfile(c), f"{c} is not a file."

    log_writer = MonitorWriter(checks[0])
    apd = []
    for k in log_writer.summaries:
        if k not in BASE_METRIC:
            apd.append(k)

    n_row = 2 + (len(apd) // 2 + len(apd) % 2)
    n_col = 2

    _, axes = plt.subplots(n_row, n_col, figsize=(
        3*n_col, 2.2*n_row), constrained_layout=True)

    if legends is None:
        legends = [str(i+1) for i in range(len(checks))]

    for clog in checks:
        log_writer = MonitorWriter(clog)
        draw_lr(axes[0][0], log_writer['train:lr'])
        draw_time(axes[0][1], log_writer['train:loss'],
                  log_writer['eval:loss'].data['cnt'], prop_box=False)
        draw_tr_loss(axes[1][0], log_writer['train:loss'])
        draw_dev_loss(axes[1][1], log_writer['eval:loss'], False)

        # custom metric
        for i, k in enumerate(apd):
            r, c = (4+i)//2, (4+i) % 2
            draw_any(axes[r][c], log_writer[k], k)

    # rm the empty subplot
    if len(apd) % 2 != 0:
        axes[-1][-1].set_axis_off()

    axes[0][0].legend(legends, fontsize=8)
    plt.suptitle(title)

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
    plt.savefig(outpath, dpi=200, facecolor="w")
    plt.close()
    return outpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str, nargs='+',
                        help="Path to the location of log file(s).")
    parser.add_argument("--title", type=str, default=None,
                        help="Configure the plotting title.")
    parser.add_argument("--legend", type=str,
                        help="Legend for two comparing figures, split by '-'. Default: 1-2")
    parser.add_argument("-o", type=str, default=None, dest="o_path",
                        help="Path of the output figure path. If not specified, saved at the directory of input log file.")
    args = parser.parse_args()

    if len(args.log) == 1:
        plot_monitor(args.log[0], title=args.title,
                     o_path=args.o_path)
    else:
        legends = args.legend
        if legends is not None:
            legends = legends.split('-')
        cmp(args.log, legends=legends, title=args.title, o_path=args.o_path)
