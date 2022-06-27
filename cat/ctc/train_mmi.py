"""Maximum mutal information training of CTC
a.k.a. the Monte-Carlo sampling based CTC-CRF

Author: Huahuan Zheng (maxwellzh@outlook.com)
"""

from . import ctc_builder
from .train import AMTrainer
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.manager import Manager
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)
from ..shared import tokenizer as tknz
from ..rnnt.train_nce import cal_wer, custom_evaluate
from ctcdecode import CTCBeamDecoder


import argparse
from typing import *

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

    manager = Manager(
        KaldiSpeechDataset,
        sortedPadCollateASR(),
        args, build_model,
        func_eval=custom_evaluate
    )

    # training
    manager.run(args)


class CTCDiscTrainer(AMTrainer):
    def __init__(self, beamdecoder: CTCBeamDecoder, ctc_weight: Optional[float] = 0., **kwargs):
        super().__init__(**kwargs)
        assert isinstance(ctc_weight, (int, float))
        assert not self.is_crf, f"mmi mode is conflict with crf"
        assert isinstance(beamdecoder, CTCBeamDecoder)
        self.attach = {
            'decoder': beamdecoder
        }
        self.weights = {
            'ctc_loss': 1+ctc_weight
        }
        self.criterion = nn.CTCLoss(reduction='none', zero_infinity=True)

    @torch.no_grad()
    def get_wer(self, feats: torch.Tensor, labels: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):
        logits, lx = self.am(feats, lx)
        logits = logits.log_softmax(dim=-1)
        bs = logits.size(0)

        labels = labels.cpu()
        ground_truth = [
            ' '.join(
                str(x)
                for x in labels[n, :ly[n]].tolist()
            )
            for n in range(bs)
        ]

        # y_samples: (N, k, L),     ly_samples: (N, k)
        y_samples, _, _, ly_samples = self.attach['decoder'].decode(
            logits.cpu())
        hypos = [
            ' '.join(str(x)
                     for x in y_samples[n, 0, :ly_samples[n, 0]].tolist())
            for n in range(bs)
        ]

        err = cal_wer(ground_truth, hypos)
        cnt_err = sum(x for x, _ in err)
        cnt_sum = sum(x for _, x in err)
        return cnt_err, cnt_sum

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
    cfg:
        mmi:
            decoder:
                beam_size: 
                kenlm: 
                tokenizer: 
                alpha: 
                beta:
                ...
            trainer:
                ...
        # basic ctc config
        ...

    """
    assert 'mmi' in cfg, f"missing 'mmi' in field:"

    # initialize beam searcher
    assert 'decoder' in cfg['mmi'], f"missing 'decoder' in field:mmi"
    decoder_cfg = cfg['mmi']['decoder']
    for s in ['beam_size', 'kenlm', 'tokenizer']:
        assert s in decoder_cfg, f"missing '{s}' in field:mmi:decoder"
    tokenizer = tknz.load(decoder_cfg['tokenizer'])
    labels = [str(i) for i in range(tokenizer.vocab_size)]
    labels[0] = '<s>'
    labels[1] = '<unk>'
    searcher = CTCBeamDecoder(
        labels=labels,
        model_path=decoder_cfg['kenlm'],
        beam_width=decoder_cfg['beam_size'],
        alpha=decoder_cfg.get('alpha', 1.),
        beta=decoder_cfg.get('beta', 0.),
        num_processes=4,
        log_probs_input=True,
        is_token_based=True
    )
    del tokenizer

    trainer_cfg = cfg['mmi'].get('trainer', {})
    encoder = ctc_builder(cfg, args, dist=False, wrapper=False)
    model = CTCDiscTrainer(searcher, am=encoder, **trainer_cfg)

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
