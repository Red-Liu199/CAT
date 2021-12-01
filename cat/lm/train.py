# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""
Language model trainer.
"""

from ..shared import Manager
from ..shared import coreutils as utils
from ..shared.manager import evaluate as default_eval
from ..shared.decoder import *
from ..shared.data import (
    CorpusDataset,
    sortedPadCollateLM
)

import gather
import math
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.eval is not None:
        if ngpus_per_node > 1:
            utils.distprint(
                "worlsize > 1 might lead to incorrect ppl calculation.", args.gpu)
        args.trset = args.eval
        args.devset = args.eval
        args.batch_size = dist.get_world_size()
        if args.resume is None:
            utils.distprint(
                "You're trying to evalute over the data with untrained model.", args.gpu)

    manager = Manager(CorpusDataset, sortedPadCollateLM(),
                      args, build_model, func_eval=evaluate)

    if args.eval is not None:
        manager.model.eval()
        ppl = evaluate(manager.valloader, args, manager)
        utils.distprint(f"Perplexity over dataset is {ppl:.2f}", args.gpu)
        return

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
    return math.exp(default_eval(*args))


def build_model(args, configuration, dist=True, wrapper=True) -> LMTrainer:
    def _build_decoder(config) -> nn.Module:
        LMNet = eval(config['type'])    # type: AbsDecoder
        NetKwargs = config['kwargs']
        return LMNet(**NetKwargs)

    assert 'decoder' in configuration

    decoder = _build_decoder(configuration['decoder'])

    if wrapper:
        model = LMTrainer(decoder)
    else:
        model = decoder

    if not dist:
        return model

    # make batchnorm synced across all processes
    model = utils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


def LMParser():
    parser = utils.BasicDDPParser('Language model training.')
    parser.add_argument("--eval", type=str,
                        help="Do evaluation and calculate the PPL.")
    return parser


def main(args: argparse = None):
    if args is None:
        parser = LMParser()
        args = parser.parse_args()

    utils.setPath(args)
    utils.main_spawner(args, main_worker)
