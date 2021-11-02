"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (maxwellzh@outlook.com)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

from ..shared import Manager
from ..shared import coreutils as utils
from ..shared.manager import evaluate as default_eval
from ..shared.decoder import *
from ..shared.data import (
    BalancedDistributedSampler,
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
from torch.utils.data.distributed import DistributedSampler


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    torch.cuda.set_device(gpu)

    args.rank = args.rank * ngpus_per_node + gpu

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    test_set = CorpusDataset(args.devset)
    test_sampler = DistributedSampler(test_set)

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=sortedPadCollateLM())

    manager = Manager(build_model, args, func_eval=evaluate)
    # lm training does not need specaug
    manager.specaug = None

    # get GPU info
    gpu_info = utils.gather_all_gpu_info(args.gpu)

    utils.distprint("> Model built.", args.gpu)
    utils.distprint("  Model size:{:.2f}M".format(
        utils.count_parameters(manager.model)/1e6), args.gpu)
    if args.rank == 0 and not args.debug:
        utils.gen_readme(args.dir+'/readme.md',
                         model=manager.model, gpu_info=gpu_info)

    tr_set = CorpusDataset(args.trset)
    setattr(args, 'n_steps', 0)

    if args.databalance:
        utils.distprint(
            "> Enable data balanced loading, it takes a while to initialize...", args.gpu)
        train_sampler = BalancedDistributedSampler(
            tr_set, args.batch_size, args.len_norm)
        trainloader = DataLoader(
            tr_set, batch_sampler=train_sampler,
            num_workers=args.workers, pin_memory=True,
            collate_fn=sortedPadCollateLM())
        utils.distprint(
            "> Seq length info for balanced loading generated.", args.gpu)
        args.n_steps = train_sampler.total_size//args.batch_size//args.grad_accum_fold
    else:
        train_sampler = DistributedSampler(tr_set)

        trainloader = DataLoader(
            tr_set, batch_size=args.batch_size//ngpus_per_node, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler, collate_fn=sortedPadCollateLM())
        args.n_steps = len(trainloader)//args.grad_accum_fold

    # training
    manager.run(train_sampler, trainloader, testloader, args)


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
        return loss


class PerplexityLoss(nn.CrossEntropyLoss):
    def __init__(self, reduction: str = 'mean', *args, **kwargs) -> None:
        super().__init__(reduction='none', *args, **kwargs)
        assert reduction in ['mean', 'sum',
                             'none'], f"unknown reduction: {reduction}"
        self.ppl_reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gather_loss = super().forward(input, target)
        gather_ppl = torch.exp(gather_loss)
        if self.ppl_reduction == 'mean':
            return gather_ppl.mean()
        elif self.ppl_reduction == 'sum':
            return gather_ppl.sum()
        elif self.ppl_reduction == 'none':
            return gather_ppl
        else:
            raise RuntimeError(
                f"Invalid reduction option: {self.ppl_reduction}, expected one of ['mean', 'sum', 'none'].")


@torch.no_grad()
def evaluate(testloader: DataLoader, args: argparse.Namespace, manager: Manager):

    avg_ce_loss = default_eval(testloader, args, manager)
    return math.exp(avg_ce_loss)


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


def main():
    parser = utils.BasicDDPParser()

    args = parser.parse_args()

    utils.setPath(args)

    utils.main_spawner(args, main_worker)
