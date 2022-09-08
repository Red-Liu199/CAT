"""Sampling based SMBR training for CTC

Author: Huahuan Zheng (maxwellzh@outlook.com)
"""

from . import ctc_builder
from .train import (
    AMTrainer,
    build_beamdecoder,
    main_worker as basic_worker
)
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..rnnt.train_nce import custom_evaluate

import ctc_align
import argparse
from typing import *
from torch_edit_distance import levenshtein_distance

import torch


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    basic_worker(
        gpu, ngpus_per_node, args,
        func_build_model=build_model,
        func_eval=custom_evaluate
    )


class SMBRTrainer(AMTrainer):
    def __init__(
            self,
            n_samples: int = 256,
            **kwargs):
        super().__init__(**kwargs)
        assert isinstance(n_samples, int) and n_samples > 0
        assert not self.is_crf, f"smbr mode is conflict with crf"

        self.n_samples = n_samples
        self.register_buffer('dummy_sep', torch.tensor(
            [], dtype=torch.long), persistent=False)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):

        logits, lx = self.am(feats, lx)
        logits = logits.log_softmax(dim=-1)
        device = logits.device

        N, T, V = logits.shape
        K = self.n_samples

        # (N, T, K)
        orin_pis = torch.multinomial(
            logits.exp().view(-1, V),
            K, replacement=True
        ).view(N, T, K)

        # get alignment seqs
        # (N, T, K) -> (N, K, T) -> (N*K, T)
        ysamples = orin_pis.transpose(1, 2).contiguous().view(-1, T)

        # in-place modifed (N*K, T) -> (N*K, U)
        lx = lx.to(device=device, dtype=torch.int)
        ysamples, lsamples = ctc_align.align_(
            ysamples,
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1)
        )

        # log Q(pi|X): (N, T, K)
        q = torch.gather(logits, dim=-1, index=orin_pis)
        q *= (torch.arange(q.size(1), device=device)[
            None, :] < lx[:, None])[:, :, None]
        # (N, T, K) -> (N, K)
        q = q.sum(dim=1)

        ly = ly.to(device=device, dtype=torch.int)
        # (\sum Ui, ) -> [(U0, ), ...] -> (N, U) -> (N*K, U)
        yref = (coreutils.pad_list(torch.split(labels, ly.cpu().tolist()))
                .unsqueeze(1)
                .expand(-1, K, -1)
                .contiguous()
                .view(N*K, -1)
                )

        # count of error words
        # dis: (N*K, 4) [ins, del, sub, len]
        dis = levenshtein_distance(
            ysamples,
            yref,
            lsamples,
            ly.unsqueeze(1).repeat(1, K).contiguous().view(-1),
            self.dummy_sep,
            self.dummy_sep
        )
        penalty = torch.sum(dis[..., :3], dim=1, dtype=q.dtype)

        # est: log[P(Y|X)*R(Y, Y^*)]
        # to avoid penalty = 0, add an extra dummy error count
        est = (penalty+1).log_() + q.view(-1)
        return est.mean(dim=0)


def build_model(cfg: dict, args: argparse.Namespace) -> Union[AbsEncoder, SMBRTrainer]:
    """
    cfg:
        trainer:
            n_samples:
            ...
        # basic ctc config
        ...
    """
    assert 'trainer' in cfg, f"missing 'trainer' in field:"

    trainer_cfg = cfg['trainer']

    ctc_kwargs = {}
    ctc_kwargs['decoder'] = build_beamdecoder(
        cfg['trainer']['decoder']
    )
    ctc_kwargs['am'] = ctc_builder(cfg, args, dist=False, wrapper=False)

    model = SMBRTrainer(
        n_samples=trainer_cfg.get('n_samples', 256),
        **ctc_kwargs
    )

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


def _parser():
    return coreutils.basic_trainer_parser("CTC-sMBR samplings Trainer")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()
