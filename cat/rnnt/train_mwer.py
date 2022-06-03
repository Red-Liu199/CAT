# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# MWER training of RNN-T

from .beam_search import BeamSearcher as RNNTDecoder
from .train import TransducerTrainer
from . import rnnt_builder
from cat.shared.manager import Manager
from cat.shared import coreutils
from cat.shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)


import os
import jiwer
import argparse
from typing import *
from warp_rnnt import rnnt_loss as RNNTLoss

import torch
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
        sortedPadCollateASR(),
        args, build_model
    )

    # training
    manager.run(args)


def cnt_we(gt: List[str], hy: List[str]) -> List[int]:
    def _cnt_error_word(_gt, _hy):
        measure = jiwer.compute_measures(_gt, _hy)
        return measure['substitutions'] + measure['deletions'] + measure['insertions']
    return [_cnt_error_word(_gt, _hy) for _gt, _hy in zip(gt, hy)]


class MWERTransducerTrainer(TransducerTrainer):
    def __init__(self, beamdecoder: RNNTDecoder, mle_weight: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.searcher = beamdecoder
        assert self._compact
        assert isinstance(mle_weight, float)
        # RNN-T loss weight for joint training
        self.mle_weight = mle_weight

    def werloss(self, enc_out: torch.Tensor, frame_lens: torch.Tensor, targets: torch.Tensor, target_lens: torch.Tensor):
        """
        encoder_out : (N, T, V)
        frame_lens  : (N, )
        targets     : (N, L)
        target_lens : (N, )
        """
        device = enc_out.device

        # batched: (batchsize*beamsize, )
        targets = targets.to(
            device='cpu', dtype=torch.int32, non_blocking=True)
        frame_lens = frame_lens.to(torch.int)
        target_lens = target_lens.to(torch.int)
        bs = enc_out.size(0)
        with torch.no_grad():
            self.requires_grad_(False)
            all_hypos = self.searcher.batching_rna(enc_out, frame_lens)

            cnt_hypos = enc_out.new_tensor(
                [len(list_hypos) for list_hypos in all_hypos], dtype=torch.long
            )
            # tmp_mesh:[
            #   [0, 1, 2, ...],
            #   [0, 1, 2, ...],
            #   ...
            # ]
            tmp_mesh = torch.meshgrid(
                torch.ones_like(cnt_hypos),
                torch.arange(cnt_hypos.sum(), device=device),
                indexing='ij')[1]
            tmp_cumsum = cnt_hypos.cumsum(dim=0).unsqueeze(1)
            mask_batches = tmp_mesh < tmp_cumsum
            mask_batches[1:] *= (tmp_mesh[1:] >= tmp_cumsum[:-1])
            del tmp_mesh
            del tmp_cumsum
            batched_tokens = [
                [hypo.pred for hypo in list_hypos]
                for list_hypos in all_hypos
            ]
            del all_hypos
            self.requires_grad_(True)

        enc_out_expand = torch.cat(
            [
                enc_out[n:n+1].expand(len(batched_tokens[n]), -1, -1)
                for n in range(bs)
            ], dim=0)
        enc_out_lens = torch.cat(
            [
                frame_lens[n:n+1].expand(len(batched_tokens[n]))
                for n in range(bs)
            ], dim=0)

        gt = sum([
            [' '.join(str(x) for x in targets[n, :target_lens[n]].tolist())] *
            len(batched_tokens[n])
            for n in range(bs)
        ], [])
        batched_tokens = sum(batched_tokens, [])
        hy = [
            ' '.join(str(x) for x in hyps[1:])
            for hyps in batched_tokens
        ]
        penalty = enc_out.new_tensor(cnt_we(gt, hy))
        del gt
        del hy
        pred_in = coreutils.pad_list([
            targets.new_tensor(hypo, device=device)
            for hypo in batched_tokens
        ])
        expd_target_lens = target_lens.new_tensor(
            [len(hypo)-1 for hypo in batched_tokens])
        pred_out, _ = self.predictor(
            pred_in, input_lengths=expd_target_lens+1)

        squeezed_targets = targets.new_tensor(
            sum([hypo[1:] for hypo in batched_tokens], tuple()),
            device=device
        )
        joinout, squeezed_targets, enc_out_lens, expd_target_lens = \
            self.compute_join(enc_out_expand, pred_out,
                              squeezed_targets, enc_out_lens, expd_target_lens)
        with autocast(enabled=False):
            ll = -RNNTLoss(
                joinout.float(), squeezed_targets,
                enc_out_lens,
                expd_target_lens,
                gather=True, compact=True
            )

        # den: (bs, )
        den = torch.logsumexp(ll * mask_batches, dim=1)
        # den: (bs, ) -> (bs*n_hyps, )
        den = den[torch.repeat_interleave(cnt_hypos)]
        return torch.sum((ll-den).exp() * penalty) / bs

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens: torch.LongTensor) -> torch.FloatTensor:

        enc_out, enc_out_lens = self.encoder(inputs, in_lens)

        loss_mbr = self.werloss(
            enc_out,
            enc_out_lens,
            targets,
            target_lens
        )
        if self.mle_weight == 0.:
            return loss_mbr
        else:
            pred_out = self.predictor(torch.nn.functional.pad(
                targets, (1, 0), value=self.bos_id))[0]

            joinout, targets, enc_out_lens, target_lens = self.compute_join(
                enc_out, pred_out, targets, enc_out_lens, target_lens
            )
            with autocast(enabled=False):
                loss_mle = RNNTLoss(
                    joinout.float(), targets,
                    enc_out_lens, target_lens,
                    reduction='mean', gather=True, compact=True
                )
            return loss_mbr + self.mle_weight * loss_mle


def build_model(cfg: dict, args: argparse.Namespace, dist: bool = True) -> MWERTransducerTrainer:
    """
    cfg:
        mwer:
            decoder:
                ...
            trainer:
                ...
        # basic transducer config
        ...

    """
    assert 'mwer' in cfg, f"missing 'mwer' in field:"
    assert 'decoder' in cfg['mwer'], f"missing 'decoder' in field:mwer:"

    encoder, predictor, joiner = rnnt_builder(cfg, dist=False, wrapped=False)

    rnnt_decoder = RNNTDecoder(
        predictor=predictor,
        joiner=joiner,
        **cfg['mwer']['decoder']
    )
    model = MWERTransducerTrainer(
        rnnt_decoder,
        encoder=encoder,
        predictor=predictor,
        joiner=joiner,
        **cfg['mwer']['trainer'],
        **cfg['transducer'])

    if not dist:
        return model

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


def _parser():
    return coreutils.basic_trainer_parser("MWER Transducer Training")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()
