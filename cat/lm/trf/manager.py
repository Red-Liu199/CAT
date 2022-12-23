# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""training/evaluating manager of ebm"""
from ...shared import (
    coreutils,
    tokenizer as tknz
)
from ...shared.specaug import SpecAug
from ...shared._constants import (
    F_CHECKPOINT_LIST,
    F_TRAINING_INFO
)
from ...shared.data import (
    BatchDistSampler,
    ReadBatchDataLoader,
    PipeTokenize
)
from ...shared.scheduler import (
    State,
    build_scheduler
)
from ...shared.tokenizer import (
    gen_cache_path,
    file2bin,
    bin2file
)

import os
import sys
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
from ...shared.manager import *

class TRFManager(Manager):
    def __init__(
            self,
            T_dataset: torch.utils.data.Dataset,
            collate_fn: Callable,
            args: argparse.Namespace,
            func_build_model: Callable[[dict, argparse.Namespace], Union[nn.Module, nn.parallel.DistributedDataParallel]],
            func_train: Optional[Callable] = None,
            func_eval: Optional[Callable] = None,
            _wds_hook: Callable[[wds.WebDataset], wds.WebDataset] = None):
        """Initialize the manager for training.

        _wds_hook (callable): for webdataset loading, the dataset would be loaded as
            >>> # dataset is an instance of WebDataset
            >>> dataset = _wds_hook(dataset)
        """
        super().__init__()

        coreutils.check_parser(args, [
            'rank', 'gpu', 'workers', 'trset', 'devset',
            'batching_mode', 'batching_uneven',
            'batch_size', 'grad_accum_fold', 'config', 'dir', 'debug',
            '_logdir', '_checkdir', 'resume', 'init_model'
        ])

        # setup dataloader
        val_set = T_dataset(args.devset)

        setattr(args, 'n_steps', 0)
        world_size = dist.get_world_size()
        if args.batching_mode == 'batch' and (not args.batching_uneven):
            args.batch_size = (args.batch_size//world_size) * world_size

        if args.large_dataset:
            assert args.tokenizer is not None, f"--tokenizer is required for --large-dataset"
            assert os.path.isfile(args.tokenizer), \
                f"--tokenizer={args.tokenizer} is not a valid file."
            # large dataset doesnot support dynamic dispatching
            args.batching_uneven = False
            args.batching_mode = 'batch'
            args.batch_size = (args.batch_size//world_size) * world_size

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
            os.environ['WORLD_SIZE'] = str(world_size)
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
                args.batch_size//world_size,
                collation_fn=collate_fn,
                # set partial=False to avoid a partial batch, but would drop a few of data, see bellow disscussion.
                partial=False
            )

            trainloader = wds.WebLoader(
                tr_set, num_workers=1, shuffle=False,
                # batching is done by webdataset
                batch_size=None)
            tr_sampler = None
            trainloader = ReadBatchDataLoader(trainloader, bs=args.batch_size)
        else:
            tr_set = T_dataset(args.trset)
            if args.batching_mode == 'bucket' and args.grad_accum_fold > 1:
                """
                NOTE (huahuan): with dynamic batching, in batch mode, at each update, the global
                ... batch size (g_bs) is always `args.batch_size`. However, with bucket mode,
                ... the g_bs could be different at steps, so the
                ... grad_accum_fold would introduce grad bias. I think this won't
                ... affect much, because the g_bs would only vary in a small range.
                ... That's why here is a WARNING instead of an ERROR.
                """
                coreutils.distprint(
                    "warning: bucket dynamic batching with --grad_accum_fold > 1 "
                    "would probably produce inconsistent results.",
                    args.gpu
                )

            tr_sampler = BatchDistSampler(
                dataset=tr_set,
                mode=args.batching_mode,
                dispatch_even=(not args.batching_uneven),
                global_batch_size=args.batch_size,
                max_bucket_size=args.bucket_size,
                local_rank=args.gpu
            )
            trainloader = DataLoader(
                tr_set, 
                batch_sampler=tr_sampler,
                num_workers=args.workers, 
                collate_fn=collate_fn
            )
            trainloader = ReadBatchDataLoader(trainloader)

        val_sampler = BatchDistSampler(
            dataset=val_set,
            mode=args.batching_mode,
            dispatch_even=(not args.batching_uneven),
            global_batch_size=args.batch_size,
            max_bucket_size=args.bucket_size,
            local_rank=args.gpu,
            shuffle=False
        )
        # NOTE: global batch size info is not required for evaluation.
        valloader = DataLoader(
            val_set,
            batch_sampler=val_sampler,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        self.train_sampler = tr_sampler
        self.trainloader = trainloader
        self.valloader = valloader

        # Initial model
        cfg = coreutils.readjson(args.config)  # type: dict
        self.model = func_build_model(cfg, args)

        coreutils.distprint("> model built. # of params: {:.2f} M".format(
            coreutils.count_parameters(self.model)/1e6), args.gpu)

        # get GPU info and create readme.md
        # NOTE: the following function requires the allreduce OP, so don't put it inside the `if...:` block
        gpu_info = coreutils.gather_all_gpu_info(args.gpu)
        if args.rank == 0 and not args.debug:
            coreutils.gen_readme(
                os.path.join(
                    args.dir,
                    F_TRAINING_INFO
                ),
                model=self.model,
                gpu_info=gpu_info
            )

        # hook the function
        self.train = train if func_train is None else func_train
        self.evaluate = evaluate if func_eval is None else func_eval

        # Initial specaug module
        if 'specaug' not in cfg:
            specaug = None
        else:
            specaug = SpecAug(**cfg['specaug']).to(f'cuda:{args.gpu}')
        self.specaug = specaug

        # Initial scheduler and optimizer
        assert 'scheduler' in cfg
        # self.scheduler = build_scheduler(
        #     cfg['scheduler'], self.model.parameters())
        self.scheduler_zeta = build_scheduler(
                    cfg.get('scheduler_zeta', cfg['scheduler']),
                    map(lambda x: x[1], filter(
                        lambda x: 'zeta' in x[0] and x[1].requires_grad, self.model.named_parameters()))
                )
        self.scheduler_noise = build_scheduler(
                    cfg.get('scheduler_noise', cfg['scheduler']),
                    map(lambda x: x[1], filter(
                        lambda x: 'noise' in x[0] and x[1].requires_grad, self.model.named_parameters()))
                )
        self.scheduler = build_scheduler(
                    cfg['scheduler'],
                    map(lambda x: x[1], filter(
                        lambda x: 'noise' not in x[0] and 'zeta' not in x[0] and x[1].requires_grad, 
                        self.model.named_parameters()))
        )

        # Initialize the grad scaler
        self.scaler = GradScaler(enabled=args.amp)

        self.rank = args.rank   # type: int
        self.DEBUG = args.debug  # type: bool
        self.epoch = 1      # type: int
        self.step = 0       # type: int
        # used to resume from checkpoint
        self.step_by_last_epoch = 0   # type: int

        if not (args.resume is None or args.init_model is None):
            coreutils.distprint(
                "warning: you specify both --resume and --init-model, "
                "but --init-model will be ignored.", args.rank)

        if args.resume is not None:
            coreutils.distprint(
                f"> resuming from: {args.resume}", args.gpu)
            checkpoint = torch.load(
                args.resume, map_location=f'cuda:{args.gpu}')  # type: OrderedDict
            self.load(checkpoint)
            del checkpoint
        elif args.init_model is not None:
            coreutils.distprint(
                f"> initialize model from: {args.init_model}", args.gpu)
            checkpoint = torch.load(
                args.init_model, map_location=f'cuda:{args.gpu}')  # type: OrderedDict

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

        # Initialize the checkpoint manager
        try:
            user = os.getlogin()
        except OSError:
            user = "defaultUser"

        self.cm = CheckManager(
            os.path.join(args._checkdir, F_CHECKPOINT_LIST),
            header=f"created by {user} at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Initialize the tensorboard
        if args.rank == 0:
            self.writer = SummaryWriter(os.path.join(
                args._logdir, "{0:%Y%m%d-%H%M%S/}".format(datetime.now())))
        else:
            self.writer = None



        
    def save(self, name: str, PATH: str = '') -> str:
        if self.model.module.lm.feat_disc:
            f_cache = gen_cache_path()
            with open(f_cache, 'w') as f:
                self.model.module.lm.wfeat.save(f)

            extra_states = {
                'disc_feats': file2bin(f_cache)
            }
            os.remove(f_cache)
            return super().save(name, PATH, extra_states)
        else:
            extra_states = {}
            if hasattr(self, 'scheduler_zeta'):
                if isinstance(self.scheduler_zeta.optimizer, ZeroRedundancyOptimizer):
                    self.scheduler_zeta.optimizer.consolidate_state_dict(0)
                extra_states['scheduler_zeta'] = self.scheduler_zeta.state_dict()
            if hasattr(self, 'scheduler_noise'):
                if isinstance(self.scheduler_noise.optimizer, ZeroRedundancyOptimizer):
                    self.scheduler_noise.optimizer.consolidate_state_dict(0)
                extra_states['scheduler_noise'] = self.scheduler_noise.state_dict()
            return super().save(name, PATH, extra_states)

    def load(self, checkpoint: OrderedDict):
        if self.model.module.lm.feat_disc:
            states = super().load(checkpoint, return_state=True)
            f_cache = bin2file(states['disc_feats'])
            del states
            with open(f_cache, 'r') as fi:
                self.model.module.lm.wfeats.restore(fi)
            os.remove(f_cache)
            return None
        else:
            if 'scheduler_zeta' in checkpoint:
                self.scheduler_zeta.load_state_dict(checkpoint['scheduler_zeta'])
            if 'scheduler_noise' in checkpoint:
                self.scheduler_noise.load_state_dict(checkpoint['scheduler_noise'])
            return super().load(checkpoint)

@torch.no_grad()
def evaluate_trf(testloader: DataLoader, args: argparse.Namespace, manager: Manager) -> float:
    
    model = manager.model
    cnt_seq = 0
    total_loss = 0.
    total_log_prob_trf = 0
    total_log_prob_noise = 0
    total_tokens = 0

    for i, minibatch in tqdm(enumerate(testloader), desc=f'Epoch: {manager.epoch} | eval',
                             unit='batch', total=len(testloader), disable=(args.gpu != 0), leave=False):

        feats, ilens, labels, olens = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)

        '''
        Suppose the loss is reduced by mean
        '''
        loss, _, metrics = model(feats, labels, ilens, olens)

        cnt_seq += feats.size(0)
        total_tokens += ilens.sum()
        total_loss += loss * feats.size(0)
        total_log_prob_trf += metrics['train/log_prob_trf']/1000 # To avoid overflow
        total_log_prob_noise += metrics['train/log_prob_noise']/1000

    cnt_seq = total_loss.new_tensor(cnt_seq)

    # sync info for loggin and further state control
    # NOTE: this sync is required.
    dist.all_reduce(total_loss, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_seq, dist.ReduceOp.SUM)
    avg_loss = (total_loss/cnt_seq).item()
    PPL_trf = torch.exp(-total_log_prob_trf/(total_tokens/1000)).item()
    PPL_noise = torch.exp(-total_log_prob_noise/(total_tokens/1000)).item()

    if args.rank == 0:
        manager.writer.add_scalar('loss/dev', avg_loss, manager.step)
        manager.writer.add_scalar('dev/ppl', PPL_trf, manager.step)
        manager.writer.add_scalar('dev/ppl_noise', PPL_noise, manager.step)
    return avg_loss

@torch.no_grad()
def evaluate_ebm(testloader: DataLoader, args: argparse.Namespace, manager: Manager) -> float:
    
    model = manager.model
    cnt_seq = 0
    total_loss = 0.
    total_tokens = 0
    avg_ppl = 0
    for i, minibatch in tqdm(enumerate(testloader), desc=f'Epoch: {manager.epoch} | eval',
                             unit='batch', total=len(testloader), disable=(args.gpu != 0), leave=False):

        feats, ilens, labels, olens = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)

        '''
        Suppose the loss is reduced by mean
        '''
        loss, _, metrics = model(feats, labels, ilens, olens)

        cnt_seq += feats.size(0)
        total_tokens += ilens.sum()
        total_loss += metrics['train/loss_true'] * feats.size(0)
        avg_ppl += metrics['train/ppl_data']

    cnt_seq = total_loss.new_tensor(cnt_seq)

    # sync info for loggin and further state control
    # NOTE: this sync is required.
    dist.all_reduce(total_loss, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_seq, dist.ReduceOp.SUM)
    avg_loss = (total_loss/cnt_seq).item()
    avg_ppl = (avg_ppl/len(testloader)).item()

    if args.rank == 0:
        manager.writer.add_scalar('loss/dev', avg_loss, manager.step)
        manager.writer.add_scalar('dev/ppl', avg_ppl, manager.step)
    return avg_loss

def train_trf(trainloader: ReadBatchDataLoader, args: argparse.Namespace, manager: Manager, hook_func: Callable = None):
    """
    The default train function.

    Args:
        trainloader (Dataloader)
        args (Namespace) : configurations
        manager (Manager) : the manager for pipeline control
        _trainer_hook (optional, callable function) : custom hook function, check source code for usage.
    """

    def _go_step(g_batch_size: int, minibatch) -> Tuple[torch.Tensor, int]:
        feats, frame_lens, labels, label_lens = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)
        if manager.specaug is not None:
            feats, frame_lens = manager.specaug(feats, frame_lens)

        with autocast(enabled=use_amp):
            if hook_func is None:
                loss = model(feats, labels, frame_lens, label_lens)
            else:
                # you could custom model forward, tracks logging and metric calculation in the hook
                loss = hook_func(
                    manager, model, args, i+1,
                    (feats, labels, frame_lens, label_lens)
                )
            if isinstance(loss, tuple):
                loss = loss[0]

            raw_loss = loss.detach()
            # divide loss with fold since we want the gradients to be divided by fold
            loss /= fold

        loss.data = loss.detach() * (feats.size(0) * world_size / g_batch_size)
        scaler.scale(loss).backward()

        # return for logging
        return raw_loss, feats.size(0)

    coreutils.check_parser(args, ['grad_accum_fold', 'n_steps', 'verbose',
                                  'print_freq', 'check_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm'])

    model = manager.model
    scaler = manager.scaler
    scheduler = manager.scheduler
    scheduler_z = manager.scheduler_zeta
    scheduler_n = manager.scheduler_noise
    optimizer = scheduler.optimizer
    optimizer_z = scheduler_z.optimizer
    optimizer_n = scheduler_n.optimizer
    optimizer.zero_grad()
    optimizer_z.zero_grad()
    optimizer_n.zero_grad()
    use_amp = args.amp
    grad_norm = args.grad_norm

    world_size = dist.get_world_size()
    fold = args.grad_accum_fold
    assert fold >= 1
    accum_loss = 0.
    n_batch = 0
    t_data = 0.
    t_last_step = time.time()
    t_last_batch = time.time()
    cnt_step_update = 0
    is_quit = torch.tensor(0, dtype=torch.bool, device=args.gpu)

    def get_progress_bar():
        return tqdm(
            desc=f'Epoch: {manager.epoch} | train',
            unit='batch',
            total=(args.n_steps if args.check_freq == -1 else args.check_freq),
            disable=(args.gpu != 0 or args.verbose),
            leave=False
        )
    # when check_freq > epoch size, the progress bar would display in mistake.
    p_bar = get_progress_bar()
    for i, (bs, minibatch) in enumerate(trainloader):
        # since the gradient fold could be > 1, we need to accumulate the time
        if args.verbose:
            t_data += time.time() - t_last_batch

        # skip steps when resuming from stop training
        if (cnt_step_update + manager.step_by_last_epoch < manager.step):
            if fold == 1 or (i+1) % fold == 0:
                cnt_step_update += 1
                p_bar.update()
                if args.verbose and args.gpu == 0:
                    sys.stderr.write(
                        f"\rIn skipping steps: {cnt_step_update + manager.step_by_last_epoch}/{manager.step}")
                    sys.stderr.flush()
            continue

        dist.all_reduce(is_quit, op=dist.ReduceOp.MAX)
        if is_quit:
            break

        # update every fold times and drop the last few batches (number of which <= fold)
        if fold == 1 or (i+1) % fold == 0:
            local_loss, local_bs = _go_step(bs, minibatch)
            accum_loss += local_loss * local_bs
            n_batch += local_bs

            if grad_norm > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    scaler.unscale_(optimizer_z)
                    scaler.unscale_(optimizer_n)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm, error_if_nonfinite=False)

            scaler.step(optimizer)
            scaler.step(optimizer_z)
            scaler.step(optimizer_n)
            scaler.update()
            optimizer.zero_grad()
            optimizer_z.zero_grad()
            optimizer_n.zero_grad()

            manager.step += 1
            scheduler.update_lr_step(manager.step)
            scheduler_z.update_lr_step(manager.step)
            scheduler_n.update_lr_step(manager.step)
            cnt_step_update += 1
            p_bar.update()

            # measure accuracy and record loss; item() can sync all processes.
            tolog = {
                'loss': (accum_loss/n_batch).item(),
                'lr': scheduler.lr_cur,
                'lr_z': scheduler_z.lr_cur,
                'lr_n': scheduler_n.lr_cur
            }

            # update tensorboard
            if args.rank == 0:
                manager.writer.add_scalar(
                    'loss/train_loss', tolog['loss'], manager.step)
                manager.writer.add_scalar(
                    'lr', tolog['lr'], manager.step)
                manager.writer.add_scalar(
                    'lr_z', tolog['lr_z'], manager.step)
                manager.writer.add_scalar(
                    'lr_n', tolog['lr_n'], manager.step)


            if args.verbose:
                coreutils.distprint(
                    f"[{manager.epoch} - {cnt_step_update}/{args.n_steps}] | data {t_data:6.3f} | time {time.time()-t_last_step:6.3f} | "
                    f"loss {tolog['loss']:.2e} | lr {tolog['lr']:.2e}",
                    args.gpu)
                t_data = 0.0
                t_last_step = time.time()

            if args.check_freq != -1 and (manager.step % args.check_freq) == 0:
                p_bar.close()
                yield None
                p_bar = get_progress_bar()

            # reset accumulated loss
            accum_loss = 0.
            n_batch = 0
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                local_loss, local_bs = _go_step(bs, minibatch)
                accum_loss += local_loss * local_bs
                n_batch += local_bs

        if args.verbose:
            t_last_batch = time.time()

    if not is_quit:
        # set quit flag to True
        is_quit = ~is_quit
        # wait until other processes quit
        dist.all_reduce(is_quit, op=dist.ReduceOp.MAX)

    manager.step_by_last_epoch += cnt_step_update
    # update n_steps, since we don't know how many steps there are with large dataset mode.
    args.n_steps = cnt_step_update
    if args.check_freq == -1:
        yield
    p_bar.close()
    return

def train_ebm(trainloader: ReadBatchDataLoader, args: argparse.Namespace, manager: Manager, hook_func: Callable = None):
    """
    The default train function.

    Args:
        trainloader (Dataloader)
        args (Namespace) : configurations
        manager (Manager) : the manager for pipeline control
        _trainer_hook (optional, callable function) : custom hook function, check source code for usage.
    """

    def _go_step(g_batch_size: int, minibatch) -> Tuple[torch.Tensor, int]:
        feats, frame_lens, labels, label_lens = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)
        if manager.specaug is not None:
            feats, frame_lens = manager.specaug(feats, frame_lens)

        with autocast(enabled=use_amp):
            if hook_func is None:
                loss = model(feats, labels, frame_lens, label_lens)
            else:
                # you could custom model forward, tracks logging and metric calculation in the hook
                loss = hook_func(
                    manager, model, args, i+1,
                    (feats, labels, frame_lens, label_lens)
                )
            if isinstance(loss, tuple):
                loss = loss[0]

            raw_loss = loss.detach()
            # divide loss with fold since we want the gradients to be divided by fold
            loss /= fold

        loss.data = loss.detach() * (feats.size(0) * world_size / g_batch_size)
        scaler.scale(loss).backward()

        # return for logging
        return raw_loss, feats.size(0)

    coreutils.check_parser(args, ['grad_accum_fold', 'n_steps', 'verbose',
                                  'print_freq', 'check_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm'])

    model = manager.model
    scaler = manager.scaler
    scheduler = manager.scheduler
    optimizer = scheduler.optimizer
    optimizer.zero_grad()
    use_amp = args.amp
    grad_norm = args.grad_norm

    world_size = dist.get_world_size()
    fold = args.grad_accum_fold
    assert fold >= 1
    accum_loss = 0.
    n_batch = 0
    t_data = 0.
    t_last_step = time.time()
    t_last_batch = time.time()
    cnt_step_update = 0
    is_quit = torch.tensor(0, dtype=torch.bool, device=args.gpu)

    def get_progress_bar():
        return tqdm(
            desc=f'Epoch: {manager.epoch} | train',
            unit='batch',
            total=(args.n_steps if args.check_freq == -1 else args.check_freq),
            disable=(args.gpu != 0 or args.verbose),
            leave=False
        )
    # when check_freq > epoch size, the progress bar would display in mistake.
    p_bar = get_progress_bar()
    for i, (bs, minibatch) in enumerate(trainloader):
        # since the gradient fold could be > 1, we need to accumulate the time
        if args.verbose:
            t_data += time.time() - t_last_batch

        # skip steps when resuming from stop training
        if (cnt_step_update + manager.step_by_last_epoch < manager.step):
            if fold == 1 or (i+1) % fold == 0:
                cnt_step_update += 1
                p_bar.update()
                if args.verbose and args.gpu == 0:
                    sys.stderr.write(
                        f"\rIn skipping steps: {cnt_step_update + manager.step_by_last_epoch}/{manager.step}")
                    sys.stderr.flush()
            continue

        dist.all_reduce(is_quit, op=dist.ReduceOp.MAX)
        if is_quit:
            break

        # update every fold times and drop the last few batches (number of which <= fold)
        if fold == 1 or (i+1) % fold == 0:
            local_loss, local_bs = _go_step(bs, minibatch)
            accum_loss += local_loss * local_bs
            n_batch += local_bs

            if grad_norm > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm, error_if_nonfinite=False)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            manager.step += 1
            if hasattr(model.module.lm, "dynamic_ratio") and model.module.lm.dynamic_ratio:
                model.module.lm.noise_mask_ratio = max(0.1, model.module.lm.noise_mask_ratio-2e-5)
            scheduler.update_lr_step(manager.step)
            cnt_step_update += 1
            p_bar.update()

            # measure accuracy and record loss; item() can sync all processes.
            tolog = {
                'loss': (accum_loss/n_batch).item(),
                'lr': scheduler.lr_cur
            }

            # update tensorboard
            if args.rank == 0:
                manager.writer.add_scalar(
                    'loss/train_loss', tolog['loss'], manager.step)
                manager.writer.add_scalar(
                    'lr', tolog['lr'], manager.step)

            if args.verbose:
                coreutils.distprint(
                    f"[{manager.epoch} - {cnt_step_update}/{args.n_steps}] | data {t_data:6.3f} | time {time.time()-t_last_step:6.3f} | "
                    f"loss {tolog['loss']:.2e} | lr {tolog['lr']:.2e}",
                    args.gpu)
                t_data = 0.0
                t_last_step = time.time()

            if args.check_freq != -1 and (manager.step % args.check_freq) == 0:
                p_bar.close()
                yield None
                p_bar = get_progress_bar()

            # reset accumulated loss
            accum_loss = 0.
            n_batch = 0
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                local_loss, local_bs = _go_step(bs, minibatch)
                accum_loss += local_loss * local_bs
                n_batch += local_bs

        if args.verbose:
            t_last_batch = time.time()

    if not is_quit:
        # set quit flag to True
        is_quit = ~is_quit
        # wait until other processes quit
        dist.all_reduce(is_quit, op=dist.ReduceOp.MAX)

    manager.step_by_last_epoch += cnt_step_update
    # update n_steps, since we don't know how many steps there are with large dataset mode.
    args.n_steps = cnt_step_update
    if args.check_freq == -1:
        yield
    p_bar.close()
    return

