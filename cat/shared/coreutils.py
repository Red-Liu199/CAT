# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""basic functions impl"""

import os
import argparse
import numpy as np
from collections import OrderedDict
from typing import Callable, Union, Sequence, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence


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
        "**This part is auto-generated, add your details in Appendix**",
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
        parser.add_argument("--amp", action="store_true",
                            help="Enable automatic mixed precision training.")
        parser.add_argument("--update-bn", action="store_true",
                            help="Update batchnorm stats for model averaging.")
        parser.add_argument("--grad-accum-fold", type=int, default=1,
                            help="Utilize gradient accumulation for K times. Default: K=1")
        parser.add_argument("--grad-norm", type=float, default=0.0,
                            help="Max norm of the gradients. Default: 0.0 (Disable grad-norm).")

        parser.add_argument("--debug", action="store_true",
                            help="Configure to debug settings, would overwrite most of the options.")
        parser.add_argument("--checkall", action="store_true",
                            help="Save all checkpoints instead only the recent one.")

        parser.add_argument("--trset", type=str, default=None,
                            help="Location of training data. Default: <data>/[pickle|hdf5]/tr.[pickle|hdf5]")
        parser.add_argument("--devset", type=str, default=None,
                            help="Location of dev data. Default: <data>/[pickle|hdf5]/cv.[pickle|hdf5]")
        parser.add_argument("--dir", type=str, default=None, metavar='PATH',
                            help="Directory to save the log and model files.")
        parser.add_argument("--databalance", action="store_true",
                            help="Load data batches according to sequence lenth.")
        parser.add_argument("--len-norm", type=str, default=None,
                            help="Normal expression to seq len. Useful with --databalance. E.g. 'L**1.3'")

    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of backbone.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:13457', type=str,
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


def main_spawner(args, _main_worker: Callable[[int, int, argparse.Namespace], None]):
    if not torch.cuda.is_available():
        highlight_msg("CPU only training is unsupported")
        return None

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"Global number of GPUs: {args.world_size}")
    mp.spawn(_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def setPath(args: argparse.Namespace):
    """
    Set args.checksdir and args.logsdir
    """

    # set checkpoint path and log files path
    if not args.debug:
        if not os.path.isdir(args.dir):
            raise RuntimeError(
                f"--dir={args.dir} is not a valid directory.")
        # ckpt -> checks
        checksdir = os.path.join(args.dir, 'checks')
        logsdir = os.path.join(args.dir, 'logs')
        os.makedirs(checksdir, exist_ok=True)
        os.makedirs(logsdir, exist_ok=True)
    else:
        highlight_msg("Debugging")
        # This is a hack, we won't read/write anything in debug mode.
        checksdir = '/'
        logsdir = os.path.join('./', 'tmp-tensorboard-logdir')

    if args.config is None:
        args.config = os.path.join(args.dir, 'config.json')

    setattr(args, 'checksdir', checksdir)
    setattr(args, 'logsdir', logsdir)
    if os.listdir(checksdir) != [] and not args.debug and args.resume is None:
        raise FileExistsError(
            f"{args.checksdir} is not empty! Refuse to run.")


def load_checkpoint(model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel], path_ckpt: str) -> torch.nn.Module:
    """Load parameters across distributed model and its checkpoint, resolve the prefix 'module.'"""
    checkpoint = torch.load(
        path_ckpt, map_location=next(model.parameters()).device)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = checkpoint['model']
    else:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            # remove the 'module.'
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model
