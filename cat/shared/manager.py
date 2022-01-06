# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""training/evaluating manager"""

from ..shared.data import BalancedDistributedSampler
from . import scheduler
from . import coreutils as utils
from ._specaug import SpecAug
from .monitor import MonitorWriter, BASE_METRIC

import os
import argparse
import json
import shutil
from collections import OrderedDict
from typing import Callable, Union, Iterable, Optional, List
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
if torch.__version__ >= '1.8.0':
    from torch.distributed.optim import ZeroRedundancyOptimizer


class Manager(object):
    def __init__(
            self,
            Dataset: torch.utils.data.Dataset,
            collate_fn: Callable,
            args: argparse.Namespace,
            func_build_model: Callable[[argparse.Namespace, dict], Union[nn.Module, nn.parallel.DistributedDataParallel]],
            func_train: Optional[Callable] = None,
            func_eval: Optional[Callable] = None,
            extra_tracks: Union[str, List[str], None] = None):
        super().__init__()

        utils.check_parser(args, ['rank', 'gpu', 'workers', 'trset', 'devset', 'databalance',
                                  'batch_size', 'grad_accum_fold', 'config', 'dir', 'debug', 'logsdir', 'resume'])

        # setup dataloader
        tr_set = Dataset(args.trset)
        val_set = Dataset(args.devset)

        setattr(args, 'n_steps', 0)
        if args.databalance:
            utils.distprint(
                "> Enable data balanced loading\n"
                "  this takes a while for large dataset.", args.gpu)
            train_sampler = BalancedDistributedSampler(
                tr_set, args.batch_size, args.len_norm)
            trainloader = DataLoader(
                tr_set, batch_sampler=train_sampler,
                num_workers=args.workers, collate_fn=collate_fn, persistent_workers=True)
            utils.distprint(
                "> Seq length info for balanced loading generated.", args.gpu)
            args.n_steps = train_sampler.total_size//args.batch_size//args.grad_accum_fold
        else:
            train_sampler = DistributedSampler(tr_set)
            trainloader = DataLoader(
                tr_set, batch_size=args.batch_size//dist.get_world_size(), shuffle=False,
                num_workers=args.workers, sampler=train_sampler, collate_fn=collate_fn, persistent_workers=True)
            args.n_steps = len(trainloader)//args.grad_accum_fold

        val_sampler = DistributedSampler(val_set, shuffle=False)
        valloader = DataLoader(
            val_set, batch_size=args.batch_size//dist.get_world_size(), shuffle=False,
            num_workers=args.workers, sampler=val_sampler, collate_fn=collate_fn)

        self.train_sampler = train_sampler
        self.trainloader = trainloader
        self.valloader = valloader

        # Initial model
        with open(args.config, 'r') as fi:
            configures = json.load(fi)  # type: dict
        self.model = func_build_model(args, configures)

        utils.distprint("> Model built. Size: {:.2f}M".format(
            utils.count_parameters(self.model)/1e6), args.gpu)

        # get GPU info and create readme.md
        gpu_info = utils.gather_all_gpu_info(args.gpu)
        if args.rank == 0 and not args.debug:
            utils.gen_readme(os.path.join(args.dir, 'readme.md'),
                             model=self.model, gpu_info=gpu_info)

        # hook the function
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
            utils.distprint("> Disable SpecAug", args.gpu)
        else:
            specaug = SpecAug(**configures['specaug_config'])
            specaug = specaug.to(f'cuda:{args.gpu}')
        self.specaug = specaug

        # Initial scheduler and optimizer
        assert 'scheduler' in configures
        if hasattr(self.model, "requires_slice") and self.model.requires_slice:
            parameters = filter(lambda x: x.requires_grad,
                                self.model.parameters())
            self.scheduler = GetScheduler(
                configures['scheduler'], parameters)
        else:
            self.scheduler = GetScheduler(
                configures['scheduler'], self.model.parameters())

        self.monitor = MonitorWriter(args.logsdir)
        self.monitor.addWriter(BASE_METRIC)
        if extra_tracks is not None:
            self.monitor.addWriter(extra_tracks)

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
                f"> Resuming from: {args.resume}", args.gpu)
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(
                args.resume, map_location=loc)  # type: OrderedDict
            self.load(checkpoint)

    def run(self, args: argparse.Namespace):

        utils.check_parser(
            args, ['checksdir', 'rank', 'gpu', 'dir', 'checkall'])

        self.model.train()
        while True:
            self.epoch += 1
            self.train_sampler.set_epoch(self.epoch)

            self.train(self.trainloader, args, self)

            self.model.eval()
            metrics = self.evaluate(self.valloader, args, self)
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
            if self.rank == 0 and not self.DEBUG:
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
            loss = model(features, labels, input_lengths, label_lengths)

        if isinstance(loss, tuple):
            assert len(loss) == 2
            loss, norm_size = loss
        else:
            norm_size = features.size(0)
        loss = loss / fold
        if not isinstance(norm_size, torch.Tensor):
            norm_size = input_lengths.new_tensor(
                int(norm_size), device=args.gpu)
        else:
            norm_size = norm_size.to(device=args.gpu, dtype=torch.long)

        normalized_loss = loss.detach() * norm_size
        if 'databalance' in args and args.databalance:
            '''
            get current global size
            efficiently, we can set t_batch_size=args.batch_size, 
            but current impl is more robust
            '''
            dist.all_reduce(norm_size)
            loss.data = normalized_loss * (world_size / norm_size)
        else:
            norm_size *= world_size

        scaler.scale(loss).backward()

        dist.all_reduce(normalized_loss)
        detach_loss += normalized_loss.float()
        n_batch += norm_size

        return detach_loss, n_batch

    utils.check_parser(args, ['grad_accum_fold', 'n_steps',
                              'print_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm'])

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

        features, input_lengths, labels, label_lengths = minibatch
        features, labels, input_lengths, label_lengths = features.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        if manager.specaug is not None:
            features, input_lengths = manager.specaug(features, input_lengths)

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
def evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:

    model = manager.model
    cnt_seq = 0
    total_loss = 0.
    for i, minibatch in tqdm(enumerate(testloader), desc=f'Epoch {manager.epoch} | eval',
                             unit='batch', total=len(testloader), disable=(args.gpu != 0), leave=False):
        if args.debug and i >= 20:
            dist.barrier()
            break

        features, input_lengths, labels, label_lengths = minibatch
        features, labels, input_lengths, label_lengths = features.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        '''
        Suppose the loss is reduced by mean
        '''
        loss = model(features, labels, input_lengths, label_lengths)

        if isinstance(loss, tuple):
            assert len(loss) >= 2
            loss, norm_size = loss[:2]
        else:
            norm_size = features.size(0)
        if not isinstance(norm_size, torch.Tensor):
            norm_size = input_lengths.new_tensor(
                int(norm_size), device=args.gpu)
        else:
            norm_size = norm_size.to(device=args.gpu, dtype=torch.long)

        real_loss = loss * norm_size  # type: torch.Tensor

        dist.all_reduce(real_loss, dist.ReduceOp.SUM)
        dist.all_reduce(norm_size, dist.ReduceOp.SUM)

        cnt_seq += norm_size.item()
        total_loss += real_loss.item()

    avg_loss = total_loss/cnt_seq
    manager.monitor.update('eval:loss', avg_loss)

    return avg_loss
