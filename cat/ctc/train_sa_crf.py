"""Maximum mutal information training of CTC
a.k.a. the Monte-Carlo sampling based CTC-CRF

Author: Huahuan Zheng (maxwellzh@outlook.com)
"""

from . import ctc_builder
from .train import (
    AMTrainer,
    build_beamdecoder,
    main_worker as basic_worker
)
from ..lm import lm_builder
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.decoder import AbsDecoder
from ..rnnt.train_nce import custom_evaluate

import ctc_align
import argparse
from typing import *

import torch
import torch.nn as nn


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    basic_worker(
        gpu, ngpus_per_node, args,
        func_build_model=build_model,
        func_eval=custom_evaluate
    )


class SACRFTrainer(AMTrainer):
    def __init__(
            self,
            lm: AbsDecoder,
            lm_weight: float = 1.0,
            ctc_weight: float = 0.,
            n_samples: int = 256,
            **kwargs):
        super().__init__(**kwargs)
        assert isinstance(n_samples, int) and n_samples > 0
        assert isinstance(lm_weight, (int, float)) and lm_weight > 0
        assert isinstance(ctc_weight, (int, float))
        assert not self.is_crf, f"sa-crf mode is conflict with crf"

        self.attach['lm'] = lm
        self.weights = {
            'lm_weight': lm_weight,
            'ctc_loss': 1+ctc_weight
        }
        self.n_samples = n_samples
        self.criterion = nn.CTCLoss(reduction='none', zero_infinity=True)
        self._pad = nn.ConstantPad1d((1, 0), 0)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):

        logits, lx = self.am(feats, lx)
        logits = logits.log_softmax(dim=-1)
        device = logits.device
        lx = lx.to(device=device, dtype=torch.int)
        ly = ly.to(device=device, dtype=torch.int)
        labels = labels.to(torch.int)

        # numerator: (N, )
        num = self.criterion(
            logits.transpose(0, 1),
            labels.to(device='cpu'),
            lx, ly
        )
        # for evaluation, use CTC loss only
        if not self.training:
            return num.mean(dim=0)

        N, T, V = logits.shape
        K = self.n_samples

        # (N, T, K)
        orin_pis = torch.multinomial(
            logits.exp().view(-1, V),
            K, replacement=True
        ).view(N, T, K)

        # get alignment seqs
        # (N, T, K) -> (N, K, T) -> (N*K, T)
        pis = orin_pis.transpose(1, 2).contiguous().view(-1, T)

        # (N*K, T) -> (N*K, U)
        ysamples, lsamples = ctc_align.align_(
            pis,
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1)
        )

        ysamples = ysamples[:, :lsamples.max()]
        ysamples *= torch.arange(ysamples.size(1), device=device)[
            None, :] < lsamples[:, None]

        padded_targets = self._pad(ysamples)
        # <s> A B C -> A B C <s>
        dummy_targets = torch.roll(padded_targets, -1, dims=1)

        # log P(Y): (N*K, )
        p_y = self.attach['lm'].score(
            padded_targets,
            dummy_targets,
            lsamples
        ) * self.weights['lm_weight']
        # (N*K, ) -> (N, K)
        p_y = p_y.view(N, K)

        # log Q(pi|X): (N, T, K)
        q = torch.gather(logits, dim=-1, index=orin_pis)
        q *= (torch.arange(q.size(1), device=device)[
            None, :] < lx[:, None])[:, :, None]
        # (N, T, K) -> (N, K)
        q = q.sum(dim=1)

        # estimate = (P(Y_0)*log Q(pi_0|X)+...) / (P(Y_0)+...)
        # estimate = (p_y.exp()*q).sum(dim=1) / p_y.exp().sum(dim=1)
        # (N, ), a more numerical stable ver.
        estimate = torch.sum(
            q * (p_y - torch.logsumexp(p_y, dim=1, keepdim=True)).exp(), dim=1)

        return (self.weights['ctc_loss']*num + estimate).mean(dim=0)


def build_model(cfg: dict, args: argparse.Namespace) -> Union[AbsEncoder, SACRFTrainer]:
    """
    cfg:
        trainer:
            n_samples:
            ctc_weight:
            lm:
                weight:
                config:
                check:
            ...
        # basic ctc config
        ...
    """
    assert 'trainer' in cfg, f"missing 'trainer' in field:"

    trainer_cfg = cfg['trainer']
    # initialize external lm
    dummy_lm = lm_builder(coreutils.readjson(
        trainer_cfg['lm']['config']), dist=False)
    if trainer_cfg['lm'].get('check', None) is not None:
        coreutils.load_checkpoint(dummy_lm, trainer_cfg['lm']['check'])
    elm = dummy_lm.lm
    elm.eval()
    elm.requires_grad_(False)
    del dummy_lm

    # initialize beam searcher
    assert 'decoder' in trainer_cfg, f"missing 'decoder' in field:trainer"
    ctc_kwargs = {}
    if 'alpha' not in cfg['trainer']['decoder']:
        cfg['trainer']['decoder']['alpha'] = trainer_cfg['lm'].get(
            'weight', 1.0)
    if 'kenlm' not in cfg['trainer']['decoder']:
        lmconfig = coreutils.readjson(trainer_cfg['lm']['config'])
        assert lmconfig['decoder']['type'] == 'NGram', \
            f"You do not set field:trainer:decoder:kenlm and field:trainer:lm:config is not directed to a kenlm."
        cfg['trainer']['decoder']['kenlm'] = lmconfig['decoder']['kwargs']['f_binlm']

    ctc_kwargs['decoder'] = build_beamdecoder(
        cfg['trainer']['decoder']
    )
    ctc_kwargs['am'] = ctc_builder(cfg, args, dist=False, wrapper=False)

    model = SACRFTrainer(
        lm=elm,
        lm_weight=trainer_cfg['lm'].get('weight', 1.0),
        ctc_weight=trainer_cfg.get('ctc_weight', 0.0),
        n_samples=trainer_cfg.get('n_samples', 256),
        **ctc_kwargs
    )

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)
    elm.cuda(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


def _parser():
    return coreutils.basic_trainer_parser("SA-CRF CTC Trainer")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()
