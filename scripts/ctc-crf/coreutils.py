"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
"""

import os
import argparse
import time
import json
import shutil
import scheduler
import numpy as np
from collections import OrderedDict
from monitor import plot_monitor
from _specaug import SpecAug
from typing import Callable, Union, Sequence, Iterable, List, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
if torch.__version__ >= '1.8.0':
    from torch.distributed.optim import ZeroRedundancyOptimizer


class Manager(object):
    def __init__(self, func_build_model: Callable[[argparse.Namespace, dict], Union[nn.Module, nn.parallel.DistributedDataParallel]], args: argparse.Namespace):
        super().__init__()

        with open(args.config, 'r') as fi:
            configures = json.load(fi)  # type: dict

        self.model = func_build_model(args, configures)

        # Initial specaug module
        if 'specaug_config' not in configures:
            specaug = None
            if args.rank == 0:
                highlight_msg("Disable SpecAug")
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

        self.log = OrderedDict({
            'log_train': ['epoch,loss,loss_real,net_lr,time'],
            'log_eval': ['loss_real,time']
        })

        # self.writer = SummaryWriter(args.logsdir)
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
            distprint(
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

            train(trainloader, args, self)

            self.model.eval()
            metrics = test(testloader, args, self)
            if isinstance(metrics, tuple):
                # defaultly use the first one to evaluate
                metrics = metrics[0]

            if args.rank == 0:
                self.writer.add_scalar('loss/dev', metrics, self.epoch)
            state, info = self.scheduler.step(self.epoch, metrics)

            if torch.__version__ > '1.8.0' and isinstance(self.scheduler.optimizer, ZeroRedundancyOptimizer):
                self.scheduler.optimizer.consolidate_state_dict(0)

            distprint(info, args.gpu)

            self.model.train()

            if self.DEBUG:
                if state == 2:
                    distprint("Terminated: GPU[%d]" % self.rank, args.gpu)
                    dist.barrier()
                    break
                continue

            if args.checkall:
                checkpoint = os.path.join(
                    args.checksdir, "checkpoint.{:03}.pt".format(self.epoch))
            else:
                checkpoint = os.path.join(args.checksdir, "checkpoint.pt")

            if self.rank == 0:
                plot_monitor(args.dir, self.log)
                self.save(checkpoint)

            if state == 2:
                distprint("Terminated: GPU[%d]" % self.rank, args.gpu)
                dist.barrier()
                break
            elif state == 1:
                if self.rank == 0:
                    shutil.copyfile(checkpoint, os.path.join(
                        args.checksdir, "bestckpt.pt"))
                continue
            elif state == 0:
                continue
            else:
                raise RuntimeError(f"Unknown state: {state}.")

    def save(self, name: str, PATH: str = '') -> str:
        """Save checkpoint.

        The checkpoint file would be located at `PATH/name.pt`
        or `name.pt` if `PATH` is empty.
        """
        if name[-3:] != '.pt':
            name += '.pt'
        PATH = os.path.join(PATH, name)
        torch.save(OrderedDict({
            'model': self.model.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'log': OrderedDict(self.log),
            'epoch': self.epoch,
            'step': self.step
        }), PATH)
        return PATH

    def load(self, checkpoint: OrderedDict):
        r'Load checkpoint.'

        dist.barrier()
        self.model.load_state_dict(checkpoint['model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.log = checkpoint['log']

        # FIXME(huahuan): for compatible with old version, will be deprecated.
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'step' in checkpoint:
            self.step = checkpoint['step']

    def log_update(self, msg: list = [], loc: str = "log_train"):
        self.log[loc].append(msg)


def GetScheduler(scheduler_configs: dict, param_list: Iterable) -> scheduler.Scheduler:
    schdl_base = getattr(scheduler, scheduler_configs['type'])
    return schdl_base(scheduler_configs['optimizer'], param_list, **scheduler_configs['kwargs'])


def pad_list(xs: torch.Tensor, pad_value=0, dim=0) -> torch.Tensor:
    """Perform padding for the list of tensors.

    Args:
        xs (`list`): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    if dim == 0:
        return pad_sequence(xs, batch_first=True, padding_value=pad_value)
    else:
        xs = [x.transpose(0, dim) for x in xs]
        padded = pad_sequence(xs, batch_first=True, padding_value=pad_value)
        return padded.transpose(1, dim+1).contiguous()


def distprint(msg: str, gpu: int = 0, isdebug: bool = False):
    if isdebug or gpu == 0:
        print(msg)


def str2num(src: str) -> Sequence[int]:
    return list(src.encode())


def num2str(num_list: list) -> str:
    return bytes(num_list).decode()


def gather_all_gpu_info(local_gpuid: int, num_all_gpus: int = None) -> Sequence[int]:
    """Gather all gpu info based on DDP backend

    This function is supposed to be invoked in all sub-process.
    """
    if num_all_gpus is None:
        num_all_gpus = dist.get_world_size()

    gpu_info = torch.cuda.get_device_name(local_gpuid)
    gpu_info_len = torch.tensor(len(gpu_info)).cuda(local_gpuid)
    dist.all_reduce(gpu_info_len, op=dist.ReduceOp.MAX)
    gpu_info_len = gpu_info_len.cpu()
    gpu_info = gpu_info + ' ' * (gpu_info_len-len(gpu_info))

    unicode_gpu_info = torch.tensor(
        str2num(gpu_info), dtype=torch.uint8).cuda(local_gpuid)
    info_list = [torch.empty(
        gpu_info_len, dtype=torch.uint8, device=local_gpuid) for _ in range(num_all_gpus)]
    dist.all_gather(info_list, unicode_gpu_info)
    return [num2str(x.tolist()).strip() for x in info_list]


def gen_readme(path: str, model: nn.Module, gpu_info: list = []) -> str:
    if os.path.exists(path):
        return path

    model_size = count_parameters(model)/1e6

    msg = [
        "### Basic info",
        "",
        "**This part is auto generated, add your details in Appendix**",
        "",
        "* Model size/M: {:.2f}".format(model_size),
        f"* GPU info \[{len(gpu_info)}\]"
    ]
    gpu_set = list(set(gpu_info))
    gpu_set = {x: gpu_info.count(x) for x in gpu_set}
    gpu_msg = [f"  * \[{num_device}\] {device_name}" for device_name,
               num_device in gpu_set.items()]

    msg += gpu_msg + [""]
    msg += [
        "### Appendix",
        "",
        "* ",
        ""
    ]
    msg += [
        "### WER"
        "",
        "```",
        "",
        "```",
        "",
        "### Monitor figure",
        "![monitor](./monitor.png)",
        ""
    ]
    with open(path, 'w') as fo:
        fo.write('\n'.join(msg))

    return path


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def highlight_msg(msg: Union[Sequence[str], str]):
    if isinstance(msg, str):
        print("\n>>> {} <<<\n".format(msg))
        return

    try:
        terminal_col = os.get_terminal_size().columns
    except:
        terminal_col = 200
    max_len = terminal_col-4
    if max_len <= 0:
        print(msg)
        return None

    len_msg = max([len(line) for line in msg])

    if len_msg > max_len:
        len_msg = max_len
        new_msg = []
        for line in msg:
            if len(line) > max_len:
                _cur_msg = [line[i*max_len:(i+1)*max_len]
                            for i in range(len(line)//max_len+1)]
                new_msg += _cur_msg
            else:
                new_msg.append(line)
        del msg
        msg = new_msg

    for i, line in enumerate(msg):
        right_pad = len_msg-len(line)
        msg[i] = '# ' + line + right_pad*' ' + ' #'
    msg = '\n'.join(msg)

    msg = '\n' + "#"*(len_msg + 4) + '\n' + msg
    msg += '\n' + "#"*(len_msg + 4) + '\n'
    print(msg)


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
    @torch.no_grad()
    def _cal_real_loss(loss, path_weights):
        if hasattr(args, 'iscrf') and args.iscrf:
            weight = torch.sum(path_weights)
            return loss - weight
        else:
            return loss

    def _go_step(detach_loss, real_loss, n_batch):
        # we divide loss with fold since we want the gradients to be divided by fold
        with autocast(enabled=enableAMP):
            loss = model(logits, labels, input_lengths, label_lengths)/fold

        normalized_loss = loss.detach() * logits.size(0)
        if args.databalance:
            # current global size
            # efficiently, we can set t_batch_size=args.batch_size, but current impl is more robust
            t_batch_size = logits.new_tensor(logits.size(0))
            dist.all_reduce(t_batch_size)
            loss.data = normalized_loss * (world_size / t_batch_size)
        else:
            t_batch_size = logits.size(0) * world_size

        scaler.scale(loss).backward()

        detach_loss += normalized_loss.float()
        real_loss += _cal_real_loss(normalized_loss.float(), path_weights)
        n_batch += t_batch_size

        return detach_loss, real_loss, n_batch

    for attr in ['grad_accum_fold', 'n_steps', 'print_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm']:
        assert hasattr(args, attr)

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
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3e')
    if fold > 1:
        # FIXME(Huahuan): it's difficult to monitor data loading time with fold > 1
        data_time = None
        progress = ProgressMeter(
            args.n_steps,
            [batch_time, losses],
            prefix="Epoch: [{}]".format(manager.epoch))
    else:
        data_time = AverageMeter('Data', ':6.3f')
        progress = ProgressMeter(
            args.n_steps,
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(manager.epoch))

    end = time.time()
    detach_loss = 0.0
    real_loss = 0.0
    n_batch = 0
    for i, minibatch in enumerate(trainloader):
        # measure data loading time
        logits, input_lengths, labels, label_lengths, path_weights = minibatch
        logits, labels, input_lengths, label_lengths = logits.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths
        path_weights = path_weights.cuda(args.gpu, non_blocking=True)

        ########## DEBUG CODE ###########
        # print(args.rank, logits.size(0),
        #       torch.cuda.max_memory_allocated(args.rank)/1e6)
        # torch.cuda.reset_peak_memory_stats(args.rank)
        #################################

        if manager.specaug is not None:
            logits, input_lengths = manager.specaug(logits, input_lengths)

        # update every fold times and drop the last few batches (number of which <= fold)
        if fold == 1 or (i+1) % fold == 0:
            if fold == 1:
                data_time.update(time.time() - end)

            detach_loss, real_loss, n_batch = _go_step(
                detach_loss, real_loss, n_batch)

            if grad_norm > 0.0:
                if enableAMP:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm, error_if_nonfinite=True)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            manager.step += 1
            global_step = manager.step
            scheduler.update_lr(global_step)

            # average for logging
            dist.all_reduce(detach_loss)
            dist.all_reduce(real_loss)
            detach_loss /= n_batch
            real_loss /= n_batch
            # measure accuracy and record loss; item() can sync all processes.
            tolog = {
                'loss': detach_loss.item(),
                'loss_real': real_loss.item(),
                'time': time.time()-end,
                'lr': scheduler.lr_cur
            }
            end = time.time()
            losses.update(tolog['loss_real'])
            # measure elapsed time
            batch_time.update(tolog['time'])

            # update tensorboard
            if args.rank == 0:
                manager.writer.add_scalar(
                    'loss/train_loss', tolog['loss'], global_step)
                manager.writer.add_scalar(
                    'loss/train_real_loss', tolog['loss_real'], global_step)
                manager.writer.add_scalar(
                    'lr', tolog['lr'], global_step)
            # update log
            manager.log_update(
                [manager.epoch, tolog['loss'], tolog['loss_real'], tolog['lr'], tolog['time']], loc='log_train')

            n_time = (i+1)//fold
            if (n_time % args.print_freq == 0 or args.debug) and args.gpu == 0:
                progress.display(n_time)

            if args.debug and n_time >= 20:
                if args.gpu == 0:
                    highlight_msg("In debugging, quit loop")
                dist.barrier()
                break
            if n_time == args.n_steps:
                break
            # reset accumulated loss
            detach_loss -= detach_loss
            real_loss -= real_loss
            n_batch -= n_batch
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                detach_loss, real_loss, n_batch = _go_step(
                    detach_loss, real_loss, n_batch)


@torch.no_grad()
def test(testloader, args: argparse.Namespace, manager: Manager) -> float:
    def _cal_real_loss(loss, path_weights):
        if hasattr(args, 'iscrf') and args.iscrf:
            weight = torch.sum(path_weights)
            return loss - weight
        else:
            return loss

    model = manager.model

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3e')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses],
        prefix='Test: ')

    beg = time.time()
    end = time.time()
    N = 0
    SumLoss = 0.
    for i, minibatch in enumerate(testloader):
        if args.debug and i >= 20:
            if args.gpu == 0:
                highlight_msg("In debugging, quit loop")
            dist.barrier()
            break
        # measure data loading time
        logits, input_lengths, labels, label_lengths, path_weights = minibatch
        logits, labels, input_lengths, label_lengths = logits.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths
        path_weights = path_weights.cuda(args.gpu, non_blocking=True)

        '''
        Suppose model can deal with train/eval mode.
        And in eval mode, the loss (metric) is sum overall batches.
        '''
        loss = model(logits, labels, input_lengths, label_lengths)

        real_loss = _cal_real_loss(loss, path_weights)  # type: torch.Tensor
        n_batch = real_loss.new_tensor(logits.size(0), dtype=torch.long)

        dist.all_reduce(real_loss, dist.ReduceOp.SUM)
        dist.all_reduce(n_batch, dist.ReduceOp.SUM)

        # measure accuracy and record loss
        losses.update((real_loss/n_batch).item())
        N += n_batch.item()
        SumLoss += real_loss.item()

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if ((i+1) % args.print_freq == 0 or args.debug) and args.gpu == 0:
            progress.display(i+1)

    avgloss = SumLoss/N
    manager.log_update(
        [avgloss, time.time() - beg], loc='log_eval')

    return avgloss


def BasicDDPParser(istraining: bool = True, prog: str = '') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    if istraining:
        parser.add_argument('-p', '--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Distributed Data Parallel')
        parser.add_argument("--seed", type=int, default=0,
                            help="Manual seed.")
        parser.add_argument("--grad-accum-fold", type=int, default=1,
                            help="Utilize gradient accumulation for K times. Default: K=1")
        parser.add_argument("--grad-norm", type=float, default=0.0,
                            help="Max norm of the gradients. Default: 0.0 (Disable grad-norm).")

        parser.add_argument("--debug", action="store_true",
                            help="Configure to debug settings, would overwrite most of the options.")
        parser.add_argument("--checkall", action="store_true",
                            help="Save all checkpoints instead only the recent one.")

        parser.add_argument("--data", type=str, default=None,
                            help="Location of training/testing data.")
        parser.add_argument("--trset", type=str, default=None,
                            help="Location of training data. Default: <data>/[pickle|hdf5]/tr.[pickle|hdf5]")
        parser.add_argument("--devset", type=str, default=None,
                            help="Location of dev data. Default: <data>/[pickle|hdf5]/cv.[pickle|hdf5]")
        parser.add_argument("--dir", type=str, default=None, metavar='PATH',
                            help="Directory to save the log and model files.")

    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of backbone.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:12947', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    return parser


def SetRandomSeed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_syncBatchNorm(model: nn.Module) -> nn.Module:
    """Convert the BatchNorm*D in model to be sync Batch Norm
        such that it can sync across DDP processes.
    """
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


def group_by_lens(src_l: List[Any], linfo: List[int], N: int, _norm: Union[None, str] = None, consider_padding: bool = False) -> List[List[Any]]:
    """Split `src_l` by `linfo` into `N` parts.
    The split is done by a kind of greedy method, considering
    balancing the sum of lengths in each part and their paddings.
    Assume src_l is sorted by descending order.
    """
    def get_largest(_linfo, _K: int) -> List[int]:
        _len_linfo = len(_linfo)
        assert _len_linfo >= _K
        if _len_linfo == _K:
            return [0, 1]
        if _K == 1:
            return [0, _len_linfo]

        _avg = sum(_linfo)/_K
        cnt_interval = 0
        for i, l in enumerate(_linfo):
            cnt_interval += l
            if cnt_interval >= _avg:
                return [0, i+1]

    def get_smallest(_linfo, _K: int) -> List[int]:
        _len_linfo = len(_linfo)
        assert _len_linfo >= _K
        if _len_linfo == _K:
            return [_len_linfo-1, _len_linfo]
        if _K == 1:
            return [0, _len_linfo]

        _avg = sum(_linfo)/_K
        lower_bound = _len_linfo
        cnt_interval = 0
        for i in range(_len_linfo-1, -1, -1):
            cnt_interval += _linfo[i]
            if cnt_interval > _avg:
                return [lower_bound, _len_linfo]
            lower_bound -= 1

    len_src = len(linfo)
    assert len_src >= N, f"list to be split is shorter than number of groups: {len_src} < {N}"
    assert len_src == len(src_l)

    if N == 1:
        return [src_l]
    if N == len_src:
        return [[x] for x in src_l]

    if _norm is not None:
        # such as 'L**1.5'
        def norm_func(L): return eval(_norm)   # type: Callable[[int], float]
        linfo = [norm_func(x) for x in linfo]

    # greedy not optimal
    g_avg = sum(linfo) / N
    cnt_parts = 0
    res = N
    indices_fwd = [0]
    indices_bwd = [len_src]
    sliced_info = linfo[:]
    while sliced_info != []:
        running_sum = sum(linfo[indices_fwd[-1]:indices_bwd[-1]])
        running_avg = running_sum/res
        if running_avg > g_avg:
            _, sliced_idx_1 = get_largest(sliced_info, res)
            indices_fwd.append(indices_fwd[-1] + sliced_idx_1)
        else:
            sliced_idx_0, sliced_idx_1 = get_smallest(
                sliced_info, res)
            indices_bwd.append(
                sliced_idx_0 + (indices_bwd[-1] - sliced_idx_1))

        sliced_info = linfo[indices_fwd[-1]:indices_bwd[-1]]
        res -= 1

    assert indices_fwd[-1] == indices_bwd[-1]

    indices = indices_fwd[:-1]+indices_bwd[::-1]

    return [src_l[indices[i]:indices[i+1]] for i in range(N)]


def test_memory(args, manager: Manager, dataset: torch.utils.data.Dataset, collect_fn, fix_batch=False):
    """Test the batch size.

    WARNING: This function would modify the model and optimizer, 
             do not invoke this function with any train of eval process.

    Return largest batch size, which is power of 2.
    As for BalancedDistributedSampler, this function would also
    ...registers a best length normalization formula.
    """
    import math
    from torch.utils.data import DataLoader
    from dataset import BalancedDistributedSampler

    class BalancedWrapper(BalancedDistributedSampler):

        def set_batch(self, N: int):
            self.g_batch = N

        def set_norm(self, norm: str):
            self._l_norm = norm

        def __iter__(self):
            # get `batch_size` longest seqs in dataset.
            Ls = np.asarray(self._lens)
            topN = np.argpartition(
                Ls, self.g_batch)[-self.g_batch:]  # type: np.ndarray

            batches = sorted(
                topN.tolist(), key=lambda i: self._lens[i], reverse=True)

            batches = group_by_lens(
                batches, [self._lens[i] for i in batches], self.num_replicas, _norm=self._l_norm)
            # make it more balanced with gradient accumulation
            partial_indices = [batches[self.rank]]
            return iter(partial_indices)

    def test_step(loader, model, optimizer):
        scaler = GradScaler(enabled=enableAMP)
        for logits, input_lengths, labels, label_lengths, _ in loader:
            logits, labels, input_lengths, label_lengths = logits.cuda(
                args.gpu, non_blocking=True), labels, input_lengths, label_lengths

            # we divide loss with fold since we want the gradients to be divided by fold
            with autocast(enabled=enableAMP):
                loss = model(
                    logits, labels, input_lengths, label_lengths)

            normalized_loss = loss.detach() * logits.size(0)

            t_batch_size = logits.new_tensor(logits.size(0))
            dist.all_reduce(t_batch_size)
            loss.data = normalized_loss * (world_size / t_batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(args.rank, logits.size(0), end=' ')

    for attr in ['batch_size', 'amp', 'rank', 'gpu', 'databalance']:
        assert hasattr(args, attr)

    assert args.databalance
    enableAMP = args.amp
    world_size = dist.get_world_size()
    wrapped_sampler = BalancedWrapper(dataset, world_size)

    N = 2**math.ceil(math.log2(world_size))
    N_limit = 0

    # search over batch size
    wrapped_sampler.set_norm(None)
    OOM = [None for _ in range(world_size)]
    OOM_rank = False
    while not fix_batch:
        distprint(
            f"> Current limit: {N_limit}, try batch size: {N}", args.rank)
        try:
            wrapped_sampler.set_batch(N)
            dummyloader = DataLoader(
                dataset, batch_sampler=wrapped_sampler,
                num_workers=1, collate_fn=collect_fn)
            test_step(dummyloader, manager.model, manager.scheduler.optimizer)
            mem = torch.cuda.max_memory_allocated(args.gpu)
            mem_list = [None for _ in range(world_size)]
            dist.all_gather_object(mem_list, mem)
            distprint('  ' + '  '.join(["GPU[{}]: {:.1f}MB".format(i, m/1e6)
                      for i, m in enumerate(mem_list)]), args.rank)
        except RuntimeError as err:
            if 'out of memory' in str(err):
                print(err)
            else:
                raise err
            OOM_rank = True
        N_limit = N
        N *= 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dist.barrier()
        dist.all_gather_object(OOM, OOM_rank)
        if any(OOM):
            for p in manager.model.parameters():
                if p.grad is not None:
                    del p.grad
            break

    if fix_batch:
        N_limit = args.batch_size

    # search over l norm
    factor = 1.00
    distprint("> Batch size for searching l norm set to:", N_limit)
    wrapped_sampler.set_batch(N_limit)
    OOM = [None for _ in range(world_size)]
    OOM_rank = False
    mem_dict = {}
    while factor <= 2.0:
        l_norm = f'L**{factor:.2f}'
        distprint(
            f"> Try norm '{l_norm}'", args.rank)
        try:
            wrapped_sampler.set_norm(l_norm)
            dummyloader = DataLoader(
                dataset, batch_sampler=wrapped_sampler,
                num_workers=1, collate_fn=collect_fn)
            test_step(dummyloader, manager.model, manager.scheduler.optimizer)
            mem = torch.cuda.max_memory_allocated(args.gpu)
            mem_list = [None for _ in range(world_size)]
            dist.all_gather_object(mem_list, mem)
            mem_dict[l_norm] = max(mem_list)
            distprint('  ' + '  '.join(["GPU[{}]: {:.1f}MB".format(i, m/1e6)
                      for i, m in enumerate(mem_list)]), args.rank)
        except RuntimeError as err:
            if 'out of memory' in str(err):
                print(err)
            else:
                raise err
            OOM_rank = True
        dist.barrier()
        dist.all_gather_object(OOM, OOM_rank)
        factor += 0.05
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if OOM:
            for p in manager.model.parameters():
                if p.grad is not None:
                    del p.grad

    if mem_dict is {}:
        l_norm = 'L'
    else:
        l_norm = min(mem_dict.keys(), key=lambda k: mem_dict[k])
    distprint(
        f"> Recommend configure: batch size {N_limit} with norm formula '{l_norm}'", args.rank)
