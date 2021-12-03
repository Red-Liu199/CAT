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
from ..shared.data import SpeechDatasetPickle, sortedPadCollate

import os
import argparse
from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
import torch.distributed as dist


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    manager = Manager(SpeechDatasetPickle,
                      sortedPadCollate(), args, build_model)

    # training
    manager.run(args)


class AMTrainer(nn.Module):
    def __init__(self, am: model_zoo.AbsEncoder):
        super().__init__()

        self.am = am
        self.criterion = nn.CTCLoss(reduction='none')

    def forward(self, logits, labels, input_lengths, label_lengths):

        netout, lens_o = self.am(logits, input_lengths)
        lens_o = lens_o.to(torch.long)
        netout = torch.log_softmax(netout, dim=-1)

        # [N, T, C] -> [T, N, C]
        netout = netout.transpose(0, 1)
        loss = self.criterion(netout, labels, lens_o, label_lengths)

        return loss.mean()


def build_model(args: argparse.Namespace, configuration: dict, dist: bool = True, wrapper: bool = True) -> Union[model_zoo.AbsEncoder, AMTrainer]:

    assert 'encoder' in configuration
    netconfigs = configuration['encoder']
    net_kwargs = netconfigs['kwargs']   # type:dict

    am_model = getattr(model_zoo, netconfigs['type'])(
        **net_kwargs)  # type: model_zoo.AbsEncoder
    if not wrapper:
        return am_model

    model = AMTrainer(am_model)
    if not dist:
        return model

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    return model


def CTCParser():
    parser = utils.BasicDDPParser("CTC trainer.")
    return parser


def main(args: argparse.Namespace = None):
    if args is None:
        parser = CTCParser()
        args = parser.parse_args()

    utils.setPath(args)
    utils.main_spawner(args, main_worker)
