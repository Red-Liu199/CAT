"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

import coreutils
import os
import argparse
import numpy as np
import model as model_zoo
import dataset as DataSet
from collections import OrderedDict
from ctc_crf import CTC_CRF_LOSS as CRFLoss
from ctc_crf import WARP_CTC_LOSS as CTCLoss

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import ctc_crf_base


def main(args):
    if not torch.cuda.is_available():
        coreutils.highlight_msg("CPU only training is unsupported")
        return None

    ckptpath = os.path.join(args.dir, 'ckpt')
    os.makedirs(ckptpath, exist_ok=True)
    setattr(args, 'ckptpath', ckptpath)
    if os.listdir(args.ckptpath) != [] and not args.debug and args.resume is None:
        raise FileExistsError(
            f"{args.ckptpath} is not empty! Refuse to run the experiment.")

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"Global number of GPUs: {args.world_size}")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.batch_size = args.batch_size // ngpus_per_node

    if args.gpu == 0:
        print("> Data prepare")
    if args.h5py:
        data_format = "hdf5"
        coreutils.highlight_msg(
            "H5py reading might cause error with Multi-GPUs")
        Dataset = DataSet.SpeechDataset
        if args.trset is None or args.devset is None:
            raise FileNotFoundError(
                "With '--hdf5' option, you must specify data location with '--trset' and '--devset'.")
    else:
        data_format = "pickle"
        Dataset = DataSet.SpeechDatasetPickle

    if args.trset is None:
        args.trset = os.path.join(args.data, f'{data_format}/tr.{data_format}')
    if args.devset is None:
        args.devset = os.path.join(
            args.data, f'{data_format}/cv.{data_format}')

    tr_set = Dataset(args.trset)
    test_set = Dataset(args.devset)
    if args.gpu == 0:
        print("  Data prepared.")

    train_sampler = DistributedSampler(tr_set)
    test_sampler = DistributedSampler(test_set)
    test_sampler.set_epoch(1)

    trainloader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, collate_fn=DataSet.sortedPadCollate())

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=DataSet.sortedPadCollate())

    logger = OrderedDict({
        'log_train': ['epoch,loss,loss_real,net_lr,time'],
        'log_eval': ['loss_real,time']
    })
    manager = coreutils.Manager(logger, build_model, args)

    # get GPU info
    gpu_info = coreutils.gather_all_gpu_info(args.gpu)

    if args.rank == 0:
        print("> Model built.")
        print("  Model size:{:.2f}M".format(
            coreutils.count_parameters(manager.model)/1e6))

        coreutils.gen_readme(args.dir+'/readme.md',
                             model=manager.model, gpu_info=gpu_info)

    # init ctc-crf, args.iscrf is set in build_model
    if args.iscrf:
        gpus = torch.IntTensor([args.gpu])
        ctc_crf_base.init_env(f"{args.data}/den_meta/den_lm.fst", gpus)

    # training
    manager.run(train_sampler, trainloader, testloader, args)

    if args.iscrf:
        ctc_crf_base.release_env(gpus)


class AMTrainer(nn.Module):
    def __init__(self, am: nn.Module, criterion: nn.Module):
        super().__init__()

        self.am = am
        self.criterion = criterion

    def forward(self, logits, labels, input_lengths, label_lengths):
        labels = labels.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        netout, lens_o = self.infer(logits, input_lengths)
        netout = torch.log_softmax(netout, dim=-1)

        loss = self.loss_fn(netout, labels, lens_o.to(
            torch.int32).cpu(), label_lengths)

        return loss


def build_model(args, configuration, train=True) -> nn.Module:

    netconfigs = configuration['net']
    net_kwargs = netconfigs['kwargs']   # type:dict
    net = getattr(model_zoo, netconfigs['type'])

    am_model = net(**net_kwargs)    # type:nn.Module
    if not train:
        return am_model

    if 'lossfn' not in netconfigs:
        lossfn = 'crf'
        coreutils.highlight_msg([
            "Warning: not specified \'lossfn\' in configuration",
            "Defaultly set to \'crf\'"
        ])
    else:
        lossfn = netconfigs['lossfn']

    if lossfn == 'crf':
        if 'lamb' not in netconfigs:
            lamb = 0.01
            coreutils.highlight_msg([
                "Warning: not specified \'lamb\' in configuration",
                "Defaultly set to 0.01"
            ])
        else:
            lamb = netconfigs['lamb']
        loss_fn = CRFLoss(lamb=lamb)
    elif lossfn == "ctc":
        loss_fn = CTCLoss()
    else:
        raise ValueError(f"Unknown loss function: {lossfn}")

    setattr(args, 'iscrf', lossfn == 'crf')
    model = AMTrainer(am_model, loss_fn)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recognition argument")

    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Distributed Data Parallel')

    parser.add_argument("--seed", type=int, default=0,
                        help="Manual seed.")

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument("--debug", action="store_true",
                        help="Configure to debug settings, would overwrite most of the options.")
    parser.add_argument("--h5py", action="store_true",
                        help="Load data with H5py, defaultly use pickle (recommended).")

    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of training procedure.")

    parser.add_argument("--data", type=str, default=None,
                        help="Location of training/testing data.")
    parser.add_argument("--trset", type=str, default=None,
                        help="Location of training data. Default: <data>/[pickle|hdf5]/tr.[pickle|hdf5]")
    parser.add_argument("--devset", type=str, default=None,
                        help="Location of dev data. Default: <data>/[pickle|hdf5]/cv.[pickle|hdf5]")
    parser.add_argument("--dir", type=str, default=None, metavar='PATH',
                        help="Directory to save the log and model files.")
    parser.add_argument("--grad-accum-fold", type=int, default=1,
                        help="Utilize gradient accumulation for K times. Default: K=1")

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:13943', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    if args.debug:
        coreutils.highlight_msg("Debugging")

    main(args)