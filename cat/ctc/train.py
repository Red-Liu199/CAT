# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

__all__ = ["AMTrainer", "build_model", "_parser", "main"]

from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)

import os
import argparse
from typing import *

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

# NOTE:
#   1/4 subsampling is used for Conformer model defaultly
#   for other sampling ratios, you may need to modify the values
#   commonly, you can use a larger value for allowing some margin.
SUBSAMPLING = 4


def check_label_len_for_ctc(tupled_mat_label: Tuple[torch.FloatTensor, torch.LongTensor]):
    """filter the short seqs for CTC/CRF"""
    return (tupled_mat_label[0].shape[0] // SUBSAMPLING > tupled_mat_label[1].shape[0])


def filter_hook(dataset):
    return dataset.select(check_label_len_for_ctc)


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
        args,
        func_build_model=build_model,
        _wds_hook=filter_hook
    )

    # NOTE: for CTC training, the input feat len must be longer than the label len
    #       ... when using webdataset (--largedataset) to load the data, we deal with
    #       ... the issue by `_wds_hook`; if not, we filter the unqualified utterances
    #       ... before training start.
    tr_dataset = manager.trainloader.dl.dataset
    if isinstance(tr_dataset, KaldiSpeechDataset):
        orilen = len(tr_dataset)
        tr_dataset.filt_by_len(lambda x, y: x//SUBSAMPLING > y)
        if len(tr_dataset) < orilen:
            coreutils.distprint(
                f"warning: filtered {orilen-len(tr_dataset)} utterances.",
                args.gpu
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
            self.criterion = nn.CTCLoss()

    def register_crf_ctx(self, den_lm: Optional[str] = None):
        """Register the CRF context on model device."""
        assert self.is_crf

        from ctc_crf import CRFContext
        self._crf_ctx = CRFContext(den_lm, next(
            iter(self.am.parameters())).device.index)

    def forward(self, feats, labels, lx, ly):

        logits, lx = self.am(feats, lx)
        logits = torch.log_softmax(logits, dim=-1)

        labels = labels.cpu()
        lx = lx.cpu()
        ly = ly.cpu()
        if self.is_crf:
            assert self._crf_ctx is not None
            with autocast(enabled=False):
                loss = self.criterion(
                    logits.float(), labels.to(torch.int),
                    lx.to(torch.int), ly.to(torch.int))
        else:
            # [N, T, C] -> [T, N, C]
            logits = logits.transpose(0, 1)
            loss = self.criterion(logits, labels.to(torch.int), lx.to(
                torch.int), ly.to(torch.int))
        return loss


def build_model(
        cfg: dict,
        args: Optional[Union[argparse.Namespace, dict]] = None,
        dist: bool = True,
        wrapper: bool = True) -> Union[nn.parallel.DistributedDataParallel, AMTrainer, model_zoo.AbsEncoder]:
    """
    for ctc-crf training, you need to add extra settings in cfg:
    {
        "trainer": {
            "use_crf": true/false,
            "lamb": 0.01,
            "den-lm": "path/to/denlm"
        },
        ...
    }
    """
    if 'trainer' not in cfg:
        cfg['trainer'] = {}

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

    model = AMTrainer(am_model, **cfg['trainer'])
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
    if 'use_crf' in cfg['trainer'] and cfg['trainer']['use_crf']:
        assert 'den-lm' in cfg['trainer']
        model.register_crf_ctx(cfg['trainer']['den-lm'])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args['gpu']])
    return model


def _parser():
    parser = coreutils.basic_trainer_parser("CTC trainer.")
    return parser


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    print(
        "NOTE:\n"
        "    since we import the build_model() function in cat.ctc,\n"
        "    we should avoid calling `python -m cat.ctc.train`, instead\n"
        "    running `python -m cat.ctc`"
    )
