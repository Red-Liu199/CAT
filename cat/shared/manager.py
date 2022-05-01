# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""training/evaluating manager"""

from ..shared.data import (
    BalancedDistributedSampler,
    PipeTokenize
)
from ..shared import tokenizer as tknz
from . import coreutils
from ._specaug import SpecAug
from .scheduler import (
    State,
    build_scheduler
)
from .monitor import (
    MonitorWriter,
    BASE_METRIC
)

import os
import glob
import time
import shutil
import argparse
import webdataset as wds
from braceexpand import braceexpand
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
from typing import *

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.optim import ZeroRedundancyOptimizer

F_CHECKLIST = "checkpoint.list"


class Manager(object):
    def __init__(
            self,
            Dataset: torch.utils.data.Dataset,
            collate_fn: Callable,
            args: argparse.Namespace,
            func_build_model: Callable[[dict, argparse.Namespace], Union[nn.Module, nn.parallel.DistributedDataParallel]],
            func_train: Optional[Callable] = None,
            func_eval: Optional[Callable] = None,
            extra_tracks: Union[str, List[str], None] = None,
            _wds_hook: Callable[[wds.WebDataset], wds.WebDataset] = None):
        """Initialize the manager for training.

        _wds_hook (callable): for webdataset loading, the dataset would be loaded as
            >>> # dataset is an instance of WebDataset
            >>> dataset = _wds_hook(dataset)
        """
        super().__init__()

        coreutils.check_parser(args, ['rank', 'gpu', 'workers', 'trset', 'devset', 'databalance',
                                      'batch_size', 'grad_accum_fold', 'config', 'dir', 'debug',
                                      'logsdir', 'checksdir', 'resume', 'init_model'])

        # setup dataloader
        val_set = Dataset(args.devset)

        setattr(args, 'n_steps', 0)
        if args.large_dataset:
            assert args.tokenizer is not None, f"--tokenizer is required for --large-dataset"
            assert os.path.isfile(args.tokenizer), \
                f"--tokenizer={args.tokenizer} is not a valid file."

            '''
            NOTE (Huahuan):
            1.  ref: https://github.com/tmbdev-archive/webdataset-examples/blob/master/main-wds.py
                Explicitly setting 'RANK' and 'WORLD_SIZE' is useful for webdataset to
                recognize the DDP. Without the setting, data in all nodes are the same.
                I guess this is a bug of WebDataset.
              
            2.  In DDP, commonly, all nodes should get the same size of batches, however, 
                the data size might not be divisible to the num_nodes as well as the batch size.
                so we just drop a few of data every epoch. This won't affect much since we usually 
                train lots of epochs. And if you're concerned about that, duplicating part of the dataset to 
                allow it fitting the size is OK. But that would require knowing the size of dataset and is somewhat more complicated.
            '''
            os.environ['RANK'] = str(dist.get_rank())
            os.environ['WORLD_SIZE'] = str(dist.get_world_size())
            tr_set = (
                wds.WebDataset(
                    # expand expression first with braceexpand, then glob, e.g.
                    # "{a,b,c}/*.tar" -> ["a/*.tar", "b/*.tar", "c/*.tar"] -> ["a/1.tar", "a/2.tar", ...]
                    [f for p_expanded in braceexpand(args.trset)
                     for f in glob.glob(p_expanded)],
                    shardshuffle=True,
                    nodesplitter=wds.shardlists.split_by_node
                )
                # buffer size of shuffling
                .shuffle(2000)
                # decode the .tar file to normal data
                .decode()
                # extract data to original tuple
                .to_tuple("mat.npy", "label.txt")
                # convert raw text into tensor with tokenizer
                .map(PipeTokenize(tknz.load(args.tokenizer)))
            )
            if _wds_hook is not None:
                # add some hook if needed, e.g. filter short seqs for CTC/CRF
                tr_set = _wds_hook(tr_set)
            tr_set = tr_set.batched(
                args.batch_size//dist.get_world_size(),
                collation_fn=collate_fn,
                # set partial=False to avoid a partial batch, but would drop a few of data, see bellow disscussion.
                partial=False
            )

            args.batch_size = (args.batch_size //
                               dist.get_world_size()) * dist.get_world_size()
            trainloader = wds.WebLoader(
                tr_set, num_workers=1, shuffle=False,
                # batching is done by webdataset
                batch_size=None)
            '''
            '''

            # we don't know the size of dataset, just set a value large enough
            args.n_steps = 0
            train_sampler = None
        else:
            tr_set = Dataset(args.trset)

            if args.databalance and dist.get_world_size() > 1:
                coreutils.distprint(
                    "> Enable balanced dataloader.", args.gpu)
                train_sampler = BalancedDistributedSampler(
                    tr_set, args.batch_size, local_rank=args.gpu)
                trainloader = DataLoader(
                    tr_set, batch_sampler=train_sampler,
                    num_workers=args.workers, collate_fn=collate_fn,
                    prefetch_factor=4, persistent_workers=True)
                args.n_steps = train_sampler.total_size//args.batch_size//args.grad_accum_fold
            else:
                args.databalance = False
                train_sampler = DistributedSampler(tr_set)
                trainloader = DataLoader(
                    tr_set, batch_size=args.batch_size//dist.get_world_size(), shuffle=False,
                    num_workers=args.workers, sampler=train_sampler, collate_fn=collate_fn,
                    prefetch_factor=4, persistent_workers=True)
                args.n_steps = len(trainloader)//args.grad_accum_fold

        val_sampler = DistributedSampler(val_set, shuffle=False)
        valloader = DataLoader(
            val_set, batch_size=args.batch_size//dist.get_world_size(), shuffle=False,
            num_workers=args.workers, sampler=val_sampler, collate_fn=collate_fn)

        self.train_sampler = train_sampler
        self.trainloader = trainloader
        self.valloader = valloader

        # Initial model
        cfg = coreutils.readjson(args.config)  # type: dict
        self.model = func_build_model(cfg, args)

        coreutils.distprint("> Model built. Size: {:.2f}M".format(
            coreutils.count_parameters(self.model)/1e6), args.gpu)

        # get GPU info and create readme.md
        # NOTE: the following function requires the allreduce OP, so don't put it inside the `if...:` block
        gpu_info = coreutils.gather_all_gpu_info(args.gpu)
        if args.rank == 0 and not args.debug:
            coreutils.gen_readme(os.path.join(args.dir, 'readme.md'),
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
        if 'specaug_config' not in cfg:
            specaug = None
            coreutils.distprint("> Disable SpecAug", args.gpu)
        else:
            specaug = SpecAug(**cfg['specaug_config'])
            specaug = specaug.to(f'cuda:{args.gpu}')
        self.specaug = specaug

        # Initial scheduler and optimizer
        assert 'scheduler' in cfg
        if hasattr(self.model, "requires_slice") and self.model.requires_slice:
            self.scheduler = build_scheduler(
                cfg['scheduler'],
                filter(lambda x: x.requires_grad, self.model.parameters())
            )
        else:
            self.scheduler = build_scheduler(
                cfg['scheduler'], self.model.parameters())

        self.cm = CheckpointManager(
            os.path.join(args.checksdir, F_CHECKLIST),
            header=f"created at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
        )
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
        self.epoch = 1      # type: int
        self.step = 0       # type: int
        # use to resume from checkpoint
        self.step_by_last_epoch = 0   # type: int

        if not (args.resume is None or args.init_model is None):
            coreutils.distprint(
                "warning: you specify both --resume and --init-model, "
                "but --init-model will be ignored.", args.rank)

        if args.resume is not None:
            coreutils.distprint(
                f"> Resuming from: {args.resume}", args.gpu)
            checkpoint = torch.load(
                args.resume, map_location=f'cuda:{args.gpu}')  # type: OrderedDict
            self.load(checkpoint)
            del checkpoint
        elif args.init_model is not None:
            coreutils.distprint(
                f"> Initialize model from: {args.init_model}", args.gpu)
            checkpoint = torch.load(
                args.init_model, map_location=f'cuda:{args.gpu}')  # type: OrderedDict
            if 'scheduler' in checkpoint:
                # load the optimizer params
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'], optim_only=True)
            try:
                self.model.load_state_dict(checkpoint['model'])
            except RuntimeError as re:
                if "Error(s) in loading state_dict" in str(re):
                    self.model.load_state_dict(
                        coreutils.translate_prev_checkpoint(
                            checkpoint['model'])
                    )
                else:
                    raise RuntimeError(str(re))
            del checkpoint

    def run(self, args: argparse.Namespace):

        coreutils.check_parser(
            args, ['checksdir', 'rank', 'gpu', 'dir'])

        self.model.train()
        terminated = False
        n_eval = 0
        while not terminated:
            if self.train_sampler is None:
                pass
            else:
                self.train_sampler.set_epoch(self.epoch)

            for _ in self.train(self.trainloader, args, self):
                n_eval += 1
                self.model.eval()
                metrics = self.evaluate(self.valloader, args, self)
                if isinstance(metrics, tuple):
                    # defaultly use the first one to evaluate
                    metrics = metrics[0]

                if args.rank == 0:
                    self.writer.add_scalar('loss/dev', metrics, self.step)
                    self.monitor.update('eval:loss', (metrics, self.step))

                state = self.scheduler.step(n_eval, metrics)
                self.model.train()

                checkpoint = os.path.join(
                    args.checksdir,
                    f"checkpoint.{self.epoch}e{self.step}s.pt"
                )
                # inside self.save(), there maybe all_reduce OP, don't put it in rank==0 block.
                # we should save the checkpoint before monitor.export(), otherwise the monitor is dumped
                # ... into file and empty.
                self.save(checkpoint)
                if self.rank == 0 and not self.DEBUG:
                    self.cm.appendinfo(
                        self.epoch, self.step,
                        metrics, self.scheduler.lr_cur, checkpoint)
                    self.monitor.visualize(args.dir)
                    self.monitor.export()

                coreutils.distprint(
                    f" Epoch: {self.epoch} | Step: {self.step} | Loss: {metrics:.3e} | LR: {self.scheduler.lr_cur:.3e}",
                    args.gpu)
                if state == State.TERMINATED:
                    # backup the last checkpoint
                    if self.rank == 0 and not self.DEBUG:
                        shutil.copyfile(checkpoint, os.path.join(
                            args.checksdir, "checkpoint.pt"))
                    print("Terminated: GPU[%d]" % self.rank)
                    terminated = True
                    dist.barrier()
                    break
                elif state == State.IMPROVED:
                    # maybe do something with the best model by far
                    pass
                elif state == State.CONTINUE:
                    pass
                else:
                    raise RuntimeError(f"Unknown state: {state}.")

            self.epoch += 1

    def save(self, name: str, PATH: str = '') -> str:
        """Save checkpoint.

        The checkpoint file would be located at:
        `PATH/name.pt`, or `name(.pt)` if `PATH` is empty.
        """

        if isinstance(self.scheduler.optimizer, ZeroRedundancyOptimizer):
            # the ZeroRedundancyOptimizer shards the optimizer into processes,
            # so we need to collect them to save on the disk.
            self.scheduler.optimizer.consolidate_state_dict(0)

        if self.rank != 0 or self.DEBUG:
            return None

        if name[-3:] != '.pt':
            name += '.pt'
        PATH = os.path.join(PATH, name)
        torch.save(OrderedDict({
            'model': self.model.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            # monitor is only for backup, never load it in manager.load()
            'monitor': self.monitor.state_dict(),
            'epoch': self.epoch,
            'step': self.step,
            'step_by_last_epoch': self.step_by_last_epoch
        }), PATH)

        return PATH

    def load(self, checkpoint: OrderedDict):
        r'Load checkpoint.'

        dist.barrier()
        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError as re:
            if "Error(s) in loading state_dict" in str(re):
                self.model.load_state_dict(
                    coreutils.translate_prev_checkpoint(checkpoint['model'])
                )
            else:
                raise RuntimeError(str(re))

        self.scheduler.load_state_dict(checkpoint['scheduler'])

        # monitor is not required to load
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.step_by_last_epoch = checkpoint.get(
            'step_by_last_epoch', self.step)


'''
NOTE (Huahuan):
    with --databalance, batch size on each device might be different,
    however, torch DDP automatically make the allreduce on gradients
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


def train(trainloader, args: argparse.Namespace, manager: Manager, _trainer_hook: Callable = None):
    """
    The default train function.

    Args:
        trainloader (Dataloader)
        args (Namespace) : configurations
        manager (Manager) : the manager for pipeline control
        _trainer_hook (optional, callable function) : custom hook function, check source code for usage.
    """

    def _go_step(detach_loss, n_batch):
        # we divide loss with fold since we want the gradients to be divided by fold
        with autocast(enabled=enableAMP):
            if _trainer_hook is None:
                loss = model(features, labels, input_lengths, label_lengths)
            else:
                # you could custom model forward, tracks logging and metric calculation in the hook
                loss = _trainer_hook(
                    manager, model, args, (i+1) % fold,
                    (features, labels, input_lengths, label_lengths)
                )

            if isinstance(loss, tuple):
                assert len(loss) == 2
                loss, norm_size = loss
            else:
                norm_size = features.size(0)
            loss /= fold

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

    coreutils.check_parser(args, ['grad_accum_fold', 'n_steps', 'verbose',
                                  'print_freq', 'check_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm'])

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
    detach_loss = 0.
    n_batch = 0
    t_data = 0.
    t_last_step = time.time()
    t_last_batch = time.time()
    quit_flag = torch.tensor(0, device=args.gpu)
    cnt_step_update = 0

    p_bar = tqdm(
        desc=f'Epoch: {manager.epoch} | train',
        unit='batch',
        total=args.n_steps,
        disable=(args.gpu != 0 or args.verbose),
        leave=False
    )
    for i, minibatch in enumerate(trainloader):
        dist.all_reduce(quit_flag, op=dist.ReduceOp.MAX)
        if quit_flag:
            # update n_steps, since we don't know how many steps there are with large dataset mode.
            args.n_steps = cnt_step_update
            break

        if cnt_step_update + manager.step_by_last_epoch < manager.step:
            if fold == 1 or (i+1) % fold == 0:
                cnt_step_update += 1
                p_bar.update()
            continue

        features, input_lengths, labels, label_lengths = minibatch
        features = features.cuda(args.gpu, non_blocking=True)
        # since the gradient fold could be > 1, we need to accumulate the time
        if args.verbose:
            t_data += time.time() - t_last_batch

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
            scheduler.update_lr_step(global_step)
            cnt_step_update += 1
            p_bar.update()

            # average for logging, since we divide loss by fold for backward,
            # here we multiply fold back for logging
            detach_loss *= fold / n_batch
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
                    'train:loss': (tolog['loss'], global_step),
                    'train:lr': (tolog['lr'], global_step)
                })

            if args.verbose:
                coreutils.distprint(
                    f"[{manager.epoch} - {cnt_step_update}/{args.n_steps}] | data {t_data:6.3f} | time {time.time()-t_last_step:6.3f} | "
                    f"loss {tolog['loss']:.2e} | lr {tolog['lr']:.2e}",
                    args.gpu)
                t_data = 0.0
                t_last_step = time.time()

            if args.check_freq != -1 and (manager.step % args.check_freq) == 0:
                yield None

            # reset accumulated loss
            detach_loss -= detach_loss
            n_batch -= n_batch
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                detach_loss, n_batch = _go_step(detach_loss, n_batch)

        if args.verbose:
            t_last_batch = time.time()

    if not quit_flag:
        # set quit flag to True
        quit_flag += 1
        # wait until other processes quit
        dist.all_reduce(quit_flag, op=dist.ReduceOp.MAX)
    manager.step_by_last_epoch += cnt_step_update
    if args.check_freq == -1:
        yield
    p_bar.close()
    return


@torch.no_grad()
def evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:

    model = manager.model
    cnt_seq = 0
    total_loss = 0.

    for i, minibatch in tqdm(enumerate(testloader), desc=f'Epoch: {manager.epoch} | eval',
                             unit='batch', total=len(testloader), disable=(args.gpu != 0), leave=False):

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

    return avg_loss


class CheckpointManager:
    def __init__(self, f_checklist: str, header: str = None) -> None:

        # the checkpoint locations would be used for identification
        '''Example
        {
            '/path/to/check000.pt': {
                'epoch': 0,
                'step':  100000,
                'metric' : 12.3,
                'lr'  :  1e-5,
                'extra' : [...]
            },
            ...
        }
        '''
        self._checks = OrderedDict()  # type: OrderedDict[str, Dict]
        self._f_checklist = f_checklist

        if header is None:
            header = ''

        if os.path.exists(f_checklist):
            # ignore the new header in case overwritten or duplicated.
            self.getcontent()
        else:
            header = '\n'.join('# '+x for x in [
                "Use '#' in a new line to identify a comment",
                "Field definition:",
                "    No.epoch No.step metric(loss) LR pathtocheckpoint ...(any append info is ok);",
                "    the float numbers are saved via (1.0).hex(), use float.fromhex('...') to get original data;",
                "    the No.step is also saved as hex via hex(123), use int('...', 16) to get original data.",
                "Header info:",
                " "*4 + header.replace('\n', ' ')
            ])
            with open(f_checklist, 'w') as fit:
                fit.write(header)

    @property
    def content(self):
        return self._checks

    def getcontent(self):
        assert os.path.isfile(self._f_checklist)

        with open(self._f_checklist, 'r') as fit:
            for line in fit:
                line = line.strip()
                if line[0] == '#' or line == '':
                    # skip the comments
                    continue
                contents = line.split()
                assert len(contents) >= 5
                n_epoch, n_step, metric, lr, f_check = contents[:5]
                self._checks.update({
                    f_check: {
                        'epoch': int(n_epoch),
                        'step': int(n_step, 16),
                        'metric': float.fromhex(metric),
                        'lr': float.fromhex(lr),
                        'extra': contents[5:]
                    }
                })

    def appendinfo(self, n_epoch: int, n_step: int, metric: float, lr: float, f_check: str, *args):
        self._checks.update({
            f_check: {
                'epoch': n_epoch,
                'step': n_step,
                'metric': metric,
                'lr': lr,
                'extra': list(args)
            }
        })
        orin_text = open(self._f_checklist, 'r').read()
        try:
            with open(self._f_checklist, 'a') as fot:
                fot.write('\n')
                fot.write((" "*4).join(
                    [
                        f"{n_epoch:04}",
                        f"{n_step:#010x}",
                        f"{float(metric).hex()}",
                        f"{float(lr).hex()}",
                        f_check
                    ]+[str(x) for x in args])
                )
        except Exception as err:
            with open(self._f_checklist, 'w') as fot:
                fot.write(orin_text)
            raise RuntimeError(str(err))
