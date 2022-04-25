# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)

import os
import argparse
from collections import OrderedDict
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    coreutils.set_random_seed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    manager = Manager(
        KaldiSpeechDataset,
        sortedPadCollateASR(flatten_target=True),
        args, build_model
    )

    # training
    manager.run(args)


class AMTrainer(nn.Module):
    def __init__(
            self,
            am: model_zoo.AbsEncoder,
            use_crf: bool = False,
            lamb: Optional[float] = 0.01,
            **kwargs):
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
            loss = self.criterion(netout, labels.to(torch.int), lens_o.to(
                torch.int), label_lengths.to(torch.int))

        return loss.mean()


def build_model(
        cfg: dict,
        args: Optional[Union[argparse.Namespace, dict]] = None,
        dist: bool = True,
        wrapper: bool = True) -> Union[nn.parallel.DistributedDataParallel, AMTrainer, model_zoo.AbsEncoder]:

    if 'ctc-trainer' not in cfg:
        cfg['ctc-trainer'] = {}

    assert 'encoder' in cfg
    netconfigs = cfg['encoder']
    net_kwargs = netconfigs['kwargs']   # type:dict

    # when immigrate configure from RNN-T to CTC,
    # one usually forget to set the `with_head=True` and 'num_classes'
    if not net_kwargs.get('with_head', False):
        print("warning: 'with_head' in field:encoder:kwargs is False/not set. "
              "If you don't know what this means, set it to True.")

    if 'num_classes' not in net_kwargs:
        raise Exception("error: 'num_classes' in field:encoder:kwargs is not set. "
                        "You should specify it according to your vocab size.")

    am_model = getattr(model_zoo, netconfigs['type'])(
        **net_kwargs)  # type: model_zoo.AbsEncoder
    if not wrapper:
        return am_model

    model = AMTrainer(am_model, **cfg['ctc-trainer'])
    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    elif not isinstance(args, dict):
        raise ValueError(f"unsupport type of args: {type(args)}")

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args['gpu'])
    if 'use_crf' in cfg['ctc-trainer'] and cfg['ctc-trainer']['use_crf']:
        assert 'den-lm' in cfg['ctc-trainer']
        model.register_crf_ctx(cfg['ctc-trainer']['den-lm'])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args['gpu']])
    return model


def CTCParser():
    parser = coreutils.basic_trainer_parser("CTC trainer.")
    return parser


def main(args: argparse.Namespace = None):
    if args is None:
        parser = CTCParser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)
