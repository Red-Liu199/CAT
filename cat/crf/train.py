# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""
This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

from ..shared import Manager
from ..shared import coreutils as utils
from ..shared import encoder as model_zoo
from ..shared.data import SpeechDatasetPickle, SpeechDataset, sortedPadCollate

import argparse

import torch
import torch.nn as nn
import torch.distributed as dist


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    from ctc_crf import CRFContext

    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    dist.barrier(device_ids=[args.gpu])

    if args.h5py:
        Dataset = SpeechDataset
    else:
        Dataset = SpeechDatasetPickle

    manager = Manager(Dataset, sortedPadCollate(), args, build_model)

    # init ctc-crf, args.iscrf is set in build_model
    if args.iscrf:
        ctx = CRFContext(args.den_lm, args.gpu)

    # training
    manager.run(args)


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
    parser.add_argument("--den-lm", type=str, default=None,
                        help="Location of denominator LM.")

    args = parser.parse_args()

    utils.setPath(args)

    utils.main_spawner(args, main_worker)
