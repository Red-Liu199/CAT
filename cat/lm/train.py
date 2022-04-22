# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""
Language model trainer.
"""

from ..shared import Manager
from ..shared import coreutils
from ..shared import decoder as model_zoo
from ..shared.decoder import AbsDecoder
from ..shared.manager import evaluate as default_eval
from ..shared.data import (
    CorpusDataset,
    sortedPadCollateLM
)

import gather
import math
import argparse
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    coreutils.set_random_seed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    manager = Manager(CorpusDataset, sortedPadCollateLM(),
                      args, build_model, func_eval=evaluate)

    # lm training does not need specaug
    manager.specaug = None

    # training
    manager.run(args)


class LMTrainer(nn.Module):
    def __init__(self, lm: AbsDecoder = None):
        super().__init__()
        self.lm = lm    # type: AbsDecoder
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:

        # preds: (N, S, C)
        preds, _ = self.lm(inputs, input_lengths=input_lengths)

        # squeeze preds by concat all sentences
        # logits: (\sum{S_i}, C)
        logits = gather.cat(preds, input_lengths)

        # targets: (\sum{S_i})
        loss = self.criterion(logits, targets)
        return loss, input_lengths.sum()


@torch.no_grad()
def evaluate(*args) -> float:
    celoss = default_eval(*args)
    try:
        # NOTE: perplexity = exp(cross-entropy loss)
        # if you custom the criterion in LMTrainer,
        # please also add a custom evaluate() function like this.
        return math.exp(celoss)
    except OverflowError:
        return float('inf')


def build_model(
        cfg: dict,
        args: Optional[Union[argparse.Namespace, dict]] = None,
        dist=True, wrapper=True) -> Union[nn.parallel.DistributedDataParallel, LMTrainer, AbsDecoder]:

    assert 'decoder' in cfg
    # when immigrate configure from RNN-T to CTC,
    # one usually forget to set the `with_head=True`
    if not cfg['decoder']['kwargs'].get('with_head', False):
        print("warning: 'with_head' in field:decoder:kwargs is False/not set. "
              "If you don't know what this means, set it to True.")

    LMNet = getattr(model_zoo, cfg['decoder']['type'])  # type: AbsDecoder
    decoder = LMNet(**cfg['decoder']['kwargs'])

    if wrapper:
        model = LMTrainer(decoder)
    else:
        model = decoder

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
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args['gpu']])

    return model


def LMParser():
    parser = coreutils.basic_trainer_parser('Language model trainer.')
    return parser


def main(args: argparse = None):
    if args is None:
        parser = LMParser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)
