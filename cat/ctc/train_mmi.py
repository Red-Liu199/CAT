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
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.manager import Manager
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)
from ..rnnt.train_nce import custom_evaluate


import argparse
from typing import *

import torch
import torch.nn as nn
import torch.distributed as dist


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    basic_worker(
        gpu, ngpus_per_node, args,
        func_build_model=build_model,
        func_eval=custom_evaluate
    )


class CTCDiscTrainer(AMTrainer):
    def __init__(self, ctc_weight: Optional[float] = 0., **kwargs):
        super().__init__(**kwargs)
        assert isinstance(ctc_weight, (int, float))
        assert not self.is_crf, f"mmi mode is conflict with crf"

        self.weights = {
            'ctc_loss': 1+ctc_weight
        }
        self.criterion = nn.CTCLoss(reduction='none', zero_infinity=True)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):

        logits, lx = self.am(feats, lx)
        logits = logits.log_softmax(dim=-1)
        device = logits.device
        lx = lx.to(device=device, dtype=torch.int)
        ly = ly.to(device=device, dtype=torch.int)
        labels = labels.to(torch.int)

        # numerator
        # p_y_x: (N, )
        p_y_x = -self.criterion(
            logits.transpose(0, 1),
            labels.to(device='cpu'),
            lx, ly
        )

        # y_samples: (N, k, L) -> (N*k, L),     ly_samples: (N, k) -> (N*k,)
        y_samples, _, _, ly_samples = self.attach['decoder'].decode(
            logits.cpu())
        y_samples = y_samples.view(-1, y_samples.size(-1)).to(torch.int)
        ly_samples = ly_samples.view(-1).to(torch.int)
        padding_mask = torch.arange(y_samples.size(1))[
            None, :] < ly_samples[:, None]
        y_samples *= padding_mask

        # denominator
        # p_y_x_den: (N, k)
        k = self.attach['decoder']._beam_width
        p_y_x_den = -self.criterion(
            # logits: (N, T, V) -> (N, k*T, V) -> (N*k, T, V) -> (T, N*k, V)
            (logits
             .repeat(1, k, 1)
             .view((-1,)+logits.shape[-2:])
             .transpose(0, 1)),
            y_samples,
            # lx: (N, ) -> (N, 1) -> (N, k) -> (N*k, )
            lx.unsqueeze(1).repeat(1, k).view(-1),
            ly_samples
        ).view(-1, k)

        return (-p_y_x + p_y_x_den.mean(dim=1)/self.weights['ctc_loss']).mean(dim=0)


def build_model(cfg: dict, args: argparse.Namespace) -> Union[AbsEncoder, CTCDiscTrainer]:
    """
    cfg: please refer to `cat.ctc.train.build_model`
    """
    # initialize beam searcher
    assert 'trainer' in cfg, f"missing 'trainer' in field:"
    assert 'decoder' in cfg['trainer'], f"missing 'decoder' in field:trainer"

    cfg['trainer']['decoder'] = build_beamdecoder(
        cfg['trainer']['decoder']
    )
    cfg['trainer']['am'] = encoder
    encoder = ctc_builder(cfg, args, dist=False, wrapper=False)
    model = CTCDiscTrainer(**cfg['trainer'])

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


def _parser():
    return coreutils.basic_trainer_parser("NCE CTC Trainer")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()
