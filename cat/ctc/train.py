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
from ..shared.data import ModifiedSpeechDataset, sortedPadCollate

import os
import argparse
from collections import OrderedDict
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    manager = Manager(ModifiedSpeechDataset,
                      sortedPadCollate(), args, build_model)

    # training
    manager.run(args)


class AMTrainer(nn.Module):
    def __init__(self,
                 am: model_zoo.AbsEncoder,
                 use_crf: bool = False,
                 lamb: Optional[float] = 0.01, **kwargs):
        super().__init__()

        self.am = am
        self.is_crf = use_crf
        if use_crf:
            from ctc_crf import CTC_CRF_LOSS as CRFLoss

            self._crf_ctx = None
            self.criterion = CRFLoss(lamb=lamb)
        else:
            self.criterion = nn.CTCLoss(reduction='none')

    def register_crf_ctx(self, den_lm: Optional[str] = None):
        """Register the CRF context on model device."""
        assert self.is_crf

        from ctc_crf import CRFContext
        self._crf_ctx = CRFContext(den_lm, next(
            iter(self.am.parameters())).device.index)

    def forward(self, logits, labels, input_lengths, label_lengths):

        netout, lens_o = self.am(logits, input_lengths)
        netout = torch.log_softmax(netout, dim=-1)

        if self.is_crf:
            assert self._crf_ctx is not None
            labels = labels.cpu()
            lens_o = lens_o.cpu()
            label_lengths = label_lengths.cpu()
            with autocast(enabled=False):
                loss = self.criterion(
                    netout.float(), labels.to(torch.int),
                    lens_o.to(torch.int), label_lengths.to(torch.int))
        else:
            # [N, T, C] -> [T, N, C]
            netout = netout.transpose(0, 1)
            lens_o = lens_o.to(torch.long)
            loss = self.criterion(netout, labels, lens_o, label_lengths)

        return loss.mean()


def build_model(args: argparse.Namespace, configuration: dict, dist: bool = True, wrapper: bool = True) -> Union[model_zoo.AbsEncoder, AMTrainer]:

    if 'ctc-trainer' not in configuration:
        configuration['ctc-trainer'] = {}

    assert 'encoder' in configuration
    netconfigs = configuration['encoder']
    net_kwargs = netconfigs['kwargs']   # type:dict

    am_model = getattr(model_zoo, netconfigs['type'])(
        **net_kwargs)  # type: model_zoo.AbsEncoder
    if not wrapper:
        return am_model

    model = AMTrainer(am_model, **configuration['ctc-trainer'])
    if not dist:
        return model

    model.cuda(args.gpu)
    if 'use_crf' in configuration['ctc-trainer'] and configuration['ctc-trainer']['use_crf']:
        assert 'den-lm' in configuration['ctc-trainer']
        model.register_crf_ctx(configuration['ctc-trainer']['den-lm'])
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
