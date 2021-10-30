# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""training/evaluating manager"""

from . import scheduler
from . import coreutils as utils
from ._specaug import SpecAug
from .monitor import MonitorWriter

import os
import argparse
import time
import json
import shutil
from collections import OrderedDict
from typing import Callable, Union, Iterable, Optional
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
if torch.__version__ >= '1.8.0':
    from torch.distributed.optim import ZeroRedundancyOptimizer


class Manager(object):
    def __init__(
            self,
            func_build_model: Callable[[argparse.Namespace, dict], Union[nn.Module, nn.parallel.DistributedDataParallel]],
            args: argparse.Namespace,
            func_train: Optional[Callable] = None,
            func_eval: Optional[Callable] = None):
        super().__init__()

        with open(args.config, 'r') as fi:
            configures = json.load(fi)  # type: dict

        self.model = func_build_model(args, configures)
        if func_train is None:
            self.train = train
        else:
            self.train = func_train

        if func_eval is None:
            self.evaluate = evaluate
        else:
            self.evaluate = func_eval

        # Initial specaug module
        if 'specaug_config' not in configures:
            specaug = None
            if args.rank == 0:
                utils.highlight_msg("Disable SpecAug")
        else:
            specaug = SpecAug(**configures['specaug_config'])
            specaug = specaug.to(f'cuda:{args.gpu}')

        self.specaug = specaug

        # Initial scheduler and optimizer
        assert 'scheduler' in configures

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            _model = self.model.module
        else:
            _model = self.model

        if hasattr(_model, "requires_slice") and _model.requires_slice:
            parameters = filter(lambda x: x.requires_grad,
                                self.model.parameters())
            self.scheduler = GetScheduler(
                configures['scheduler'], parameters)
        else:
            self.scheduler = GetScheduler(
                configures['scheduler'], self.model.parameters())
        del _model

        self.monitor = MonitorWriter(args.logsdir)
        self.monitor.addWriter(['train:loss', 'train:lr', 'eval:loss'])

        if args.rank == 0:
            self.writer = SummaryWriter(os.path.join(
                args.logsdir, "{0:%Y%m%d-%H%M%S/}".format(datetime.now())))
        else:
            self.writer = None

        self.rank = args.rank   # type: int
        self.DEBUG = args.debug  # type: bool
        self.epoch = 0      # type:int
        self.step = 0       # type:int

        if args.resume is not None:
            utils.distprint(
                f"[GPU {args.rank}]: Resuming from: {args.resume}", args.gpu)
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(
                args.resume, map_location=loc)  # type: OrderedDict
            self.load(checkpoint)

    def run(self, train_sampler: torch.utils.data.distributed.DistributedSampler, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, args: argparse.Namespace):

        for attr in ['checksdir', 'rank', 'gpu', 'dir', 'checkall']:
            assert hasattr(args, attr)
        self.model.train()
        while True:
            self.epoch += 1
            train_sampler.set_epoch(self.epoch)

            self.train(trainloader, args, self)

            self.model.eval()
            metrics = self.evaluate(testloader, args, self)
            if isinstance(metrics, tuple):
                # defaultly use the first one to evaluate
                metrics = metrics[0]

            if args.rank == 0:
                self.writer.add_scalar('loss/dev', metrics, self.epoch)

            state, info = self.scheduler.step(self.epoch, metrics)

            utils.distprint(info, args.gpu)
            self.model.train()

            if args.checkall:
                checkpoint = os.path.join(
                    args.checksdir, "checkpoint.{:03}.pt".format(self.epoch))
            else:
                checkpoint = os.path.join(args.checksdir, "checkpoint.pt")

            self.save(checkpoint)
            if self.rank == 0:
                self.monitor.visualize(args.dir)
                # skip exporting, since the monitor exported with visualize() automatically.
                # self.monitor.export()

            if state == 2:
                utils.distprint("Terminated: GPU[%d]" % self.rank, args.gpu)
                dist.barrier()
                break
            elif state == 1:
                if self.rank == 0 and not self.DEBUG:
                    shutil.copyfile(checkpoint, os.path.join(
                        args.checksdir, "bestckpt.pt"))
                continue
            elif state == 0:
                continue
            else:
                raise RuntimeError(f"Unknown state: {state}.")

    def save(self, name: str, PATH: str = '') -> str:
        """Save checkpoint.

        The checkpoint file would be located at:
        `PATH/name.pt`, or `name(.pt)` if `PATH` is empty.
        """

        if torch.__version__ > '1.8.0' and isinstance(self.scheduler.optimizer, ZeroRedundancyOptimizer):
            self.scheduler.optimizer.consolidate_state_dict(0)

        if self.rank != 0 or self.DEBUG:
            return None

        if name[-3:] != '.pt':
            name += '.pt'
        PATH = os.path.join(PATH, name)
        torch.save(OrderedDict({
            'model': self.model.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'step': self.step
        }), PATH)

        return PATH

    def load(self, checkpoint: OrderedDict):
        r'Load checkpoint.'

        dist.barrier()
        self.model.load_state_dict(checkpoint['model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        # monitor is not required to load
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']


def GetScheduler(scheduler_configs: dict, param_list: Iterable) -> scheduler.Scheduler:
    schdl_base = getattr(scheduler, scheduler_configs['type'])
    return schdl_base(scheduler_configs['optimizer'], param_list, **scheduler_configs['kwargs'])


'''
NOTE (Huahuan):
    with --databalance, batch size on each device might be different,
    however, torch DDP automatically make a allreduce on gradients
    then average them by world size during backward.
    which assumes the batch sizes across devices are the same.
    To address this, we re-calculate the loss in a hack way:
        loss_normalized = sum(loss) / global_batch_size * world_size
    Currently the loss is:
        loss_current_normalized = mean_on_device(loss) / world_size
    Substitute `loss_normalized` to replace `mean_on_device(loss)`, here is
        loss_current_normalized' = sum(loss) / global_batch_size
    such that the gradient is properly computed. Be aware that this
    might cause numerical difference with float point given the fact that
        probably: (f * N) / N != f
'''


def train(trainloader, args: argparse.Namespace, manager: Manager):

    def _go_step(detach_loss, n_batch):
        # we divide loss with fold since we want the gradients to be divided by fold
        with autocast(enabled=enableAMP):
            loss = model(logits, labels, input_lengths, label_lengths)/fold

        normalized_loss = loss.detach() * logits.size(0)
        if hasattr(args, 'databalance') and args.databalance:
            # current global size
            # efficiently, we can set t_batch_size=args.batch_size, but current impl is more robust
            t_batch_size = logits.new_tensor(logits.size(0))
            dist.all_reduce(t_batch_size)
            loss.data = normalized_loss * (world_size / t_batch_size)
        else:
            t_batch_size = logits.size(0) * world_size

        scaler.scale(loss).backward()

        detach_loss += normalized_loss.float()
        n_batch += t_batch_size

        return detach_loss, n_batch

    for attr in ['grad_accum_fold', 'n_steps', 'print_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm']:
        assert hasattr(args, attr), f"{attr} not in args"

    model = manager.model
    scheduler = manager.scheduler
    optimizer = scheduler.optimizer
    optimizer.zero_grad()
    enableAMP = args.amp
    scaler = GradScaler(enabled=enableAMP)
    grad_norm = args.grad_norm

    world_size = dist.get_world_size()
    fold = args.grad_accum_fold
    assert fold >= 1
    detach_loss = 0.0
    n_batch = 0
    for i, minibatch in tqdm(enumerate(trainloader), desc=f'Epoch {manager.epoch} | train',
                             unit='batch', total=fold*args.n_steps, disable=(args.gpu != 0), leave=False):

        logits, input_lengths, labels, label_lengths = minibatch
        logits, labels, input_lengths, label_lengths = logits.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        if manager.specaug is not None:
            logits, input_lengths = manager.specaug(logits, input_lengths)

        # update every fold times and drop the last few batches (number of which <= fold)
        if fold == 1 or (i+1) % fold == 0:
            detach_loss, n_batch = _go_step(detach_loss, n_batch)

            if grad_norm > 0.0:
                if enableAMP:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm, error_if_nonfinite=False)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            manager.step += 1
            global_step = manager.step
            scheduler.update_lr(global_step)

            # average for logging
            dist.all_reduce(detach_loss)
            detach_loss /= n_batch
            # measure accuracy and record loss; item() can sync all processes.
            tolog = {
                'loss': detach_loss.item(),
                'lr': scheduler.lr_cur
            }

            # update tensorboard
            if args.rank == 0:
                manager.writer.add_scalar(
                    'loss/train_loss', tolog['loss'], global_step)
                manager.writer.add_scalar(
                    'lr', tolog['lr'], global_step)

            # update monitor
            manager.monitor.update({
                'train:loss': tolog['loss'],
                'train:lr': tolog['lr']
            })

            n_time = (i+1)//fold

            if n_time == args.n_steps or (args.debug and n_time >= 20):
                dist.barrier()
                break

            # reset accumulated loss
            detach_loss -= detach_loss
            n_batch -= n_batch
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                detach_loss, n_batch = _go_step(detach_loss, n_batch)


@torch.no_grad()
def update_bn(trainloader, args: argparse.Namespace, manager: Manager):

    for attr in ['n_steps', 'print_freq', 'rank', 'gpu', 'debug', 'amp']:
        assert hasattr(args, attr), f"{attr} not in args"

    model = manager.model
    if not hasattr(model.module, 'impl_forward'):
        raise RuntimeError(
            "--update-bn requires the model trainer has impl_forward to do pure forward computation.")

    model.train()
    enableAMP = args.amp

    for i, minibatch in tqdm(enumerate(trainloader), desc=f'Update BN',
                             unit='batch', total=args.n_steps, disable=(args.gpu != 0), leave=False):
        # measure data loading time
        logits, input_lengths, labels, label_lengths = minibatch
        logits, labels, input_lengths, label_lengths = logits.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        with autocast(enabled=enableAMP):
            _ = model.module.impl_forward(
                logits, labels, input_lengths, label_lengths)


@torch.no_grad()
def evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:

    model = manager.model
    cnt_seq = 0
    total_loss = 0.
    for i, minibatch in tqdm(enumerate(testloader), desc=f'Epoch {manager.epoch} | eval',
                             unit='batch', total=len(testloader), disable=(args.gpu != 0), leave=False):
        if args.debug and i >= 20:
            dist.barrier()
            break

        logits, input_lengths, labels, label_lengths = minibatch
        logits, labels, input_lengths, label_lengths = logits.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        '''
        Suppose model can deal with train/eval mode.
        And in eval mode, the loss (metric) is sum overall batches.
        '''
        loss = model(logits, labels, input_lengths, label_lengths)

        real_loss = loss  # type: torch.Tensor
        n_batch = real_loss.new_tensor(logits.size(0), dtype=torch.long)

        dist.all_reduce(real_loss, dist.ReduceOp.SUM)
        dist.all_reduce(n_batch, dist.ReduceOp.SUM)

        cnt_seq += n_batch.item()
        total_loss += real_loss.item()

    avg_loss = total_loss/cnt_seq
    manager.monitor.update('eval:loss', avg_loss)

    return avg_loss
