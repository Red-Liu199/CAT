# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""
This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

from ..shared import coreutils as utils
from ..shared import encoder as model_zoo
from ..shared.data import SpeechDatasetPickle, SpeechDataset, sortedPadCollate

import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    from ctc_crf import CRFContext

    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    torch.cuda.set_device(gpu)

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.batch_size = args.batch_size // ngpus_per_node

    utils.distprint("> Data prepare", args.gpu)
    if args.h5py:
        data_format = "hdf5"
        utils.highlight_msg(
            "H5py reading might cause error with Multi-GPUs")
        Dataset = SpeechDataset
        if args.trset is None or args.devset is None:
            raise FileNotFoundError(
                "With '--hdf5' option, you must specify data location with '--trset' and '--devset'.")
    else:
        data_format = "pickle"
        Dataset = SpeechDatasetPickle

    if args.trset is None:
        args.trset = os.path.join(args.data, f'{data_format}/tr.{data_format}')
    if args.devset is None:
        args.devset = os.path.join(
            args.data, f'{data_format}/cv.{data_format}')

    tr_set = Dataset(args.trset)
    test_set = Dataset(args.devset)
    utils.distprint("  Data prepared.", args.gpu)

    train_sampler = DistributedSampler(tr_set)
    test_sampler = DistributedSampler(test_set)
    test_sampler.set_epoch(1)

    trainloader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, collate_fn=sortedPadCollate())

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=sortedPadCollate())

    manager = utils.Manager(build_model, args)

    # get GPU info
    gpu_info = utils.gather_all_gpu_info(args.gpu)

    utils.distprint("> Model built.", args.gpu)
    utils.distprint("  Model size:{:.2f}M".format(
        utils.count_parameters(manager.model)/1e6), args.gpu)

    if args.rank == 0 and not args.debug:
        utils.gen_readme(args.dir+'/readme.md',
                         model=manager.model, gpu_info=gpu_info)

    # init ctc-crf, args.iscrf is set in build_model
    if args.iscrf:
        ctx = CRFContext(f"{args.data}/den_meta/den_lm.fst", args.gpu)

    # training
    manager.run(train_sampler, trainloader, testloader, args)


class AMTrainer(nn.Module):
    def __init__(self, am: nn.Module, criterion: nn.Module):
        super().__init__()

        self.am = am
        self.criterion = criterion

    def forward(self, logits, labels, input_lengths, label_lengths):
        labels = labels.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        netout, lens_o = self.am(logits, input_lengths)
        netout = torch.log_softmax(netout, dim=-1)

        loss = self.criterion(netout, labels, lens_o.to(
            torch.int32).cpu(), label_lengths)

        return loss


def build_model(args, configuration, train=True) -> nn.Module:

    from ctc_crf import CTC_CRF_LOSS as CRFLoss
    from ctc_crf import WARP_CTC_LOSS as CTCLoss
    netconfigs = configuration['net']
    net_kwargs = netconfigs['kwargs']   # type:dict
    net = getattr(model_zoo, netconfigs['type'])

    am_model = net(**net_kwargs)    # type:nn.Module
    if not train:
        return am_model

    if 'lossfn' not in netconfigs:
        lossfn = 'crf'
        utils.highlight_msg([
            "Warning: not specified \'lossfn\' in configuration",
            "Defaultly set to \'crf\'"
        ])
    else:
        lossfn = netconfigs['lossfn']

    if lossfn == 'crf':
        if 'lamb' not in netconfigs:
            lamb = 0.01
            utils.highlight_msg([
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

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    return model


def main():
    parser = utils.BasicDDPParser()
    parser.add_argument("--h5py", action="store_true",
                        help="Load data with H5py, defaultly use pickle (recommended).")
    parser.add_argument("--data", type=str, default=None,
                        help="Location of training/testing data.")

    args = parser.parse_args()

    utils.setPath(args)

    utils.main_spawner(args, main_worker)
