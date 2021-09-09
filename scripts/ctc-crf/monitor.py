"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

Directly execute: (in working directory)
    python3 ctc-crf/monitor.py <path to my exp>
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch


def plot_monitor(log_path: str = None, log: OrderedDict = None, task: str = None, interactive_show=False):
    """Plot the monitor log files

    Args:
        log_path (str, optional): directory of log files
        log (OrderedDict, optional): log files
        task (str, optional): task name (title of ploting)
        interactive_show (bool, optional): specify whether plot in interactive mode. Default False. 
    """

    if log is None:
        # read from file
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"'{log_path}' doesn't exist!")

        log = torch.load(log_path, map_location='cpu')['log']

    if task is None:
        task = ' '

    if log_path is None:
        direc = './'
    else:
        if os.path.isfile(log_path):
            direc = os.path.dirname(log_path)
        elif os.path.isdir(log_path):
            direc = log_path
        else:
            raise ValueError(
                f"log_path={log_path} is neither a directory nor a file.")

    '''
    log = OrderedDict({
            'log_train': ['epoch,loss,loss_real,net_lr,time'],
            'log_eval': ['loss_real,time']
        })
    '''

    df_train = np.array(log['log_train'][1:])
    df_train = {
        'loss': df_train[:, 1],
        'loss_real': df_train[:, 2],
        'lr': df_train[:, 3],
        'time': df_train[:, 4],
    }
    df_eval = np.array(log['log_eval'][1:])
    df_eval = {
        'loss': df_eval[:, 0],
        'time': df_eval[:, 1]
    }
    num_batches = df_train['loss'].shape[0]
    num_epochs = df_eval['loss'].shape[0]

    _, axes = plt.subplots(2, 2)

    # Time
    ax = axes[0][0]
    batch_per_epoch = num_batches//num_epochs
    accum_time = df_train['time']
    for i in range(1, len(accum_time)):
        accum_time[i] += accum_time[i-1]
        if (i + 1) % batch_per_epoch == 0:
            accum_time[i] += df_eval['time'][(i+1)//batch_per_epoch-1]
    del batch_per_epoch
    accum_time = [x/3600 for x in accum_time]
    ax.plot(accum_time)
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
    del timestr
    del speed

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('Total time / h')
    del ax

    # Learning rate
    ax = axes[0][1]
    lrs = df_train['lr']
    sim_lrs = [0]
    for i in range(1, len(lrs)):
        if lrs[i] != lrs[i-1]:
            sim_lrs += [i-1, i]
    if sim_lrs[-1] < len(lrs) - 1:
        sim_lrs.append(len(lrs)-1)

    if len(sim_lrs) > 50 or len(sim_lrs) == 1:
        ax.semilogy(lrs)
    else:
        ax.set_yscale('log')
        for i in range(len(sim_lrs)-1):
            _xs = [sim_lrs[i], sim_lrs[i+1]]
            _ys = [lrs[sim_lrs[i]], lrs[sim_lrs[i+1]]]
            if _ys[0] == _ys[1]:
                ax.plot(_xs, _ys, color="C0")
            else:
                ax.plot(_xs, _ys, ls='--', color='black', alpha=0.5)
        del _xs
        del _ys
    del sim_lrs
    del lrs
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('learning rate')
    del ax

    # Training loss and moving average
    ax = axes[1][0]
    train_loss = df_train['loss_real']
    running_mean = [train_loss[0]]
    for i in range(1, len(train_loss)):
        running_mean.append(running_mean[i-1]*0.9+0.1*train_loss[i])
    min_loss = min(train_loss)
    if min_loss <= 0.:
        # ax.set_yscale('symlog')
        ax.plot(train_loss, alpha=0.3)
        ax.plot(running_mean, color='orangered')
    else:
        ax.semilogy(train_loss, alpha=0.3)
        ax.semilogy(running_mean, color='orangered')

    del train_loss
    del running_mean
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(True, which="both", ls='--')
    ax.set_ylabel('Train set loss')
    ax.set_xlabel("Step")
    del ax

    # Dev loss
    ax = axes[1][1]
    dev_loss = df_eval['loss']
    min_loss = min(dev_loss)
    if min_loss <= 0.:
        # ax.set_yscale('symlog')
        ax.plot([i+1 for i in range(num_epochs)], dev_loss)
    else:
        ax.semilogy([i+1 for i in range(num_epochs)], dev_loss)

    ax.axhline(y=min_loss, ls='--', color='black', alpha=0.5)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    textstr = '\n'.join([
        "min={:.2f}".format(min_loss),
        f"{num_epochs} epoch"
    ])
    speed = accum_time[-1]/num_epochs
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.grid(True, which="both", ls='--')
    ax.set_ylabel('Dev set loss')
    ax.set_xlabel('Epoch')
    del ax
    del dev_loss

    # Global settings
    titles = [
        task.replace('dev_', '')
    ]
    plt.suptitle('\n'.join(titles))
    plt.tight_layout()
    if interactive_show:
        plt.show()
    else:
        outpath = os.path.join(direc, 'monitor.png')
        plt.savefig(outpath, dpi=300)
        print(f"> Saved at {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str, help="Location of log files.")
    parser.add_argument("--title", type=str, default=None,
                        help="Configure the plotting title.")
    args = parser.parse_args()

    plot_monitor(args.log, task=args.title)
