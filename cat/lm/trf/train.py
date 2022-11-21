# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)
#         Liu Haochen add parts of training TRF by NCE
"""
TRF Language model trainer.
TRF LM is not causal.
"""

__all__ = ["TRFLMTrainer", "build_model", "_parser", "main"]

from codecs import ignore_errors
from zmq import has
from ...shared import coreutils
from ...shared.decoder import AbsDecoder
from ...shared.manager import (
    Manager,
    TRFManager,
    train_ebm as origin_train_func,
    train_trf as default_train_func,
    evaluate,
    evaluate_nce
)
from ...shared.data import (
    CorpusDataset,
    sortedPadCollateLM
)
from ...shared.tokenizer import (
    gen_cache_path,
    file2bin,
    bin2file
)
from ...shared.scheduler import build_scheduler
from . import model as model_zoo

import os
import argparse
from typing import *


import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.distributed.optim import ZeroRedundancyOptimizer
import webdataset as wds


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    coreutils.set_random_seed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    
    configures = coreutils.readjson(args.config)
    mode = configures['decoder']['kwargs'].get('method', 'nce')
    model_type = configures['decoder']['type']
    extra_tracks = [
        'train/loss_data',
        'train/loss_noise'
    ]
    if model_type=='TRFLM':
        extra_tracks +=[
            'train/acc_data',
            'train/acc_noise'
        ]
    if mode == 'dnce':
        extra_tracks += [
            'train/ppl_trfM_data',
            'train/ppl_trfM_noise',
            'train/ppl_noiseM_data',
            'train/ppl_noiseM_noise',
            'train/loss_noise_kl',
            'train/loss_data_true',
            'train/loss_noise_true',
            'train/loss_true',
            'dev/ppl_trf',
            'dev/ppl_noise',
            'train/zeta_5',
            'train/zeta_15',
            'train/zeta_25'
        ]
    evaluate_func = evaluate_nce if mode=='dnce' else evaluate
    manager_cls = TRFLMTrainer if model_type=='TRFLM' else Manager
    train_func = custom_train if model_type =='TRFLM' else origin_train
    manager = manager_cls(
        CorpusDataset,
        sortedPadCollateLM(flatten_target=False),
        args, build_model,
        func_train=train_func,
        func_eval=evaluate_func,
        extra_tracks=extra_tracks
    )
    manager.trf_mode = mode

    # lm training does not need specaug
    manager.specaug = None

    # training
    manager.run(args)


# training TRF LM
class TRFLMTrainer(nn.Module):
    def __init__(self, lm: model_zoo.TRFLM):
        super().__init__()
        self.lm = lm

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens) -> torch.FloatTensor:

        # phi: (N, S, 1)
        # the log prob for the N sentences.
        energy = self.lm(inputs=inputs, targets=targets,
                      input_lengths=in_lens)
        loss, metrics = self.lm.cal_loss(inputs, energy, in_lens, targets)
        return loss, inputs.size(0), metrics


def custom_hook(
        manager: TRFManager,
        model: TRFLMTrainer,
        args: argparse.Namespace,
        n_step: int,
        nnforward_args: tuple):

    loss, _, metrics = model(*nnforward_args)
    ignore_keys = [
        'train/log_prob_noise',
        'train/log_prob_trf',
        'train/acc_data_sample',
        'train/acc_data_noise'
        ]
    if args.rank == 0:
        step_cur = manager.step_by_last_epoch + n_step

        for k, v in metrics.items():
            if k in ignore_keys:
                continue
            manager.monitor.update(k, (float(v), step_cur))
            manager.writer.add_scalar(k, float(v), step_cur)

    return loss


def custom_train(*args):
    return default_train_func(*args, hook_func=custom_hook)

def origin_train(*args):
    return origin_train_func(*args, hook_func=custom_hook)


@torch.no_grad()
def build_model(
        cfg: dict,
        args: Optional[Union[argparse.Namespace, dict]] = None,
        dist=True, wrapper=True) -> Union[nn.parallel.DistributedDataParallel, TRFLMTrainer, AbsDecoder]:

    assert 'decoder' in cfg
    # when training standalone LM,
    # one usually forget to set the `with_head=True`
    if not cfg['decoder']['kwargs'].get('with_head', True):
        print("warning: 'with_head' in field:decoder:kwargs is False.")

    LMNet = getattr(model_zoo, cfg['decoder']['type'])
    decoder = LMNet(**cfg['decoder']['kwargs'])   # type: model_zoo.TRFLM

    if wrapper:
        model = TRFLMTrainer(decoder)
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
    if hasattr(model.lm, 'noise_module'):
        model.lm.noise_module[0].cuda(args['gpu'])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args['gpu']])#, find_unused_parameters=True)

    return model


def _parser():
    parser = coreutils.basic_trainer_parser('TRF language model trainer.')
    return parser


def main(args: argparse = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()
