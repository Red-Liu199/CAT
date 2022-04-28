# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# MWER training of RNN-T

from ..shared import Manager
from ..shared import coreutils
from ..shared.decoder import AbsDecoder
from ..shared.encoder import AbsEncoder
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)
from .beam_search import BeamSearcher as RNNTDecoder
from .train import build_model as rnnt_builder
from .train import TransducerTrainer
from .joiner import (
    AbsJointNet
)

from warp_rnnt import rnnt_loss as RNNTLoss
from warp_rnnt import fused_rnnt_loss_ as RNNTFusedLoss

import os
import jiwer
import argparse
from typing import *

import torch
import torch.nn as nn
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


def cnt_we(gt: List[int], hy: List[int]) -> List[int]:
    def _cnt_error_word(_gt, _hy):
        measure = jiwer.compute_measures(_gt, _hy)
        return measure['substitutions'] + measure['deletions'] + measure['insertions']
    return [_cnt_error_word(_gt, _hy) for _gt, _hy in zip(gt, hy)]


class MWERTrainer(TransducerTrainer):
    def __init__(self, beamdecoder: RNNTDecoder, mle_weight: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.searcher = beamdecoder
        assert self._compact
        assert not self._fused, f"RNN-T+MWER doesn't support fused mode yet."
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
        # batched: (batchsize*beamsize, )
        targets = targets.to(
            device='cpu', dtype=torch.int32, non_blocking=True)
        frame_lens = frame_lens.to(torch.int)
        target_lens = target_lens.to(torch.int)
        bs = enc_out.size(0)
        with torch.no_grad():
            istraining = self.training
            self.eval()
            self.requires_grad_(False)
            all_hypos = self.searcher.batching_rna(enc_out, frame_lens)
            # filter the ground truth label
            all_hypos = [
                [hypo for hypo in list_hypos if hypo.pred[1:] !=
                    tuple(targets[n, :target_lens[n]].tolist())]
                for n, list_hypos in enumerate(all_hypos)
            ]
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
                torch.arange(cnt_hypos.sum(), device=cnt_hypos.device),
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
            self.train(istraining)

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
            targets.new_tensor(hypo, device=enc_out.device)
            for hypo in batched_tokens
        ])
        squeezed_targets = targets.new_tensor(
            sum([hypo[1:] for hypo in batched_tokens], tuple())
        )
        expd_target_lens = target_lens.new_tensor(
            [len(hypo)-1 for hypo in batched_tokens])
        pred_out, _ = self.predictor(
            pred_in, input_lengths=expd_target_lens+1)

        joinout, squeezed_targets, enc_out_lens = self.compute_join(
            enc_out_expand, pred_out, squeezed_targets, enc_out_lens, expd_target_lens)

        with autocast(enabled=False):
            nll = RNNTLoss(
                joinout.float(), squeezed_targets,
                enc_out_lens.to(device=joinout.device),
                expd_target_lens.to(device=joinout.device),
                gather=True, compact=True
            )

        # nll: (bs*n_hyps, )
        nll = (-nll).exp()
        # den: (bs, )
        den = (nll * mask_batches).sum(dim=1)
        # den: (bs, ) -> (bs*n_hyps, )
        den = den[torch.repeat_interleave(cnt_hypos)]
        return torch.sum(nll / den * penalty) / bs

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

            joinout, targets, enc_out_lens = self.compute_join(
                enc_out, pred_out, targets, enc_out_lens, target_lens
            )
            with autocast(enabled=False):
                loss_mle = RNNTLoss(
                    joinout.float(), targets.to(dtype=torch.int32),
                    enc_out_lens.to(device=joinout.device,
                                    dtype=torch.int32),
                    target_lens.to(
                        device=joinout.device, dtype=torch.int32),
                    reduction='mean', gather=True, compact=self._compact
                )
            return loss_mbr + self.mle_weight * loss_mle


def build_model(cfg: dict, args: argparse.Namespace, dist: bool = True) -> MWERTrainer:
    """
    cfg:
        MWER:
            decoder:
                ...
            trainer:
                ...
        # basic transducer config
        ...

    """
    assert 'MWER' in cfg, f"missing 'MWER' in field:"
    assert 'decoder' in cfg['MWER'], f"missing 'decoder' in field:MWER:"

    encoder, predictor, joiner = rnnt_builder(cfg, dist=False, wrapped=False)

    rnnt_decoder = RNNTDecoder(
        predictor=predictor,
        joiner=joiner,
        **cfg['MWER']['decoder']
    )
    model = MWERTrainer(
        rnnt_decoder,
        encoder=encoder,
        predictor=predictor,
        joiner=joiner,
        **cfg['MWER']['trainer'],
        **cfg['transducer'])

    if not dist:
        return model

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


if __name__ == "__main__":
    parser = coreutils.basic_trainer_parser()
    args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)
