# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# MWER training of RNN-T

from ..shared import Manager
from ..shared import coreutils
from ..shared.decoder import AbsDecoder
from ..shared.encoder import AbsEncoder
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateTransducer
)
from .beam_search import BeamSearcher as RNNTDecoder
from .train import build_model as rnnt_builder
from .joint import (
    PackedSequence,
    AbsJointNet
)

from warp_rnnt import rnnt_loss as RNNTLoss
from warp_rnnt import fused_rnnt_loss_ as RNNTFusedLoss

import os
import jiwer
import argparse
from typing import Union, List, Tuple

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
        sortedPadCollateTransducer(),
        args, build_model
    )

    # training
    manager.run(args)


def cnt_we(gt: List[int], hy: List[int]) -> List[int]:
    def _cnt_error_word(_gt, _hy):
        measure = jiwer.compute_measures(_gt, _hy)
        return measure['substitutions'] + measure['deletions'] + measure['insertions']
    return [_cnt_error_word(_gt, _hy) for _gt, _hy in zip(gt, hy)]


class DiscTransducerTrainer(nn.Module):
    def __init__(
            self,
            beamdecoder: RNNTDecoder,
            encoder: AbsEncoder,
            predictor: AbsDecoder,
            joiner: AbsJointNet,
            fused: bool = False):
        super().__init__()
        self.searcher = beamdecoder
        self.encoder = encoder
        self.decoder = predictor
        self.joint = joiner
        self.isfused = fused

    def werloss(self, encoder_out: torch.Tensor, frame_lens: torch.Tensor, targets: torch.Tensor, target_lens: torch.Tensor):
        """
        encoder_out : (N, T, V)
        frame_lens  : (N, )
        targets     : (N, L)
        target_lens : (N, )
        """
        # batched: (batchsize*beamsize, )
        with torch.no_grad():
            self.joint.requires_grad_(False)
            self.decoder.requires_grad_(False)
            all_hypos = self.searcher.batching_rna(encoder_out, frame_lens)
            cnt_hypos = encoder_out.new_tensor(
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
            self.joint.requires_grad_(True)
            self.decoder.requires_grad_(True)

        frame_lens = frame_lens.to(torch.int)
        target_lens = target_lens.to(torch.int)
        targets = targets.to(torch.int)

        bs = encoder_out.size(0)
        enc_out_expand = torch.cat(
            [
                encoder_out[n:n+1].expand(len(batched_tokens[n]), -1, -1)
                for n in range(bs)
            ], dim=0)
        enc_out_lens = torch.cat(
            [
                frame_lens[n:n+1].expand(len(batched_tokens[n]))
                for n in range(bs)
            ], dim=0)

        gt = sum([[
            ' '.join(str(x) for x in targets[n, :target_lens[n]].cpu(
            ).tolist())]*len(batched_tokens[n]) for n in range(bs)], []
        )
        batched_tokens = sum(batched_tokens, [])
        hy = [
            ' '.join(str(x) for x in hyps[1:])
            for hyps in batched_tokens
        ]
        penalty = encoder_out.new_tensor(cnt_we(gt, hy))
        del gt
        del hy
        pred_in = coreutils.pad_list([
            targets.new_tensor(hypo)
            for hypo in batched_tokens
        ])
        squeezed_targets = targets.new_tensor(
            sum([hypo[1:] for hypo in batched_tokens], tuple())
        )
        expd_target_lens = target_lens.new_tensor(
            [len(hypo)-1 for hypo in batched_tokens])
        preditor_out, _ = self.decoder(
            pred_in, input_lengths=expd_target_lens+1)

        packed_enc_out = PackedSequence(enc_out_expand, enc_out_lens)
        packed_pred_out = PackedSequence(preditor_out, expd_target_lens+1)
        if self.isfused:
            joiner_out = self.joint.impl_forward(
                packed_enc_out, packed_pred_out)
        else:
            joiner_out = self.joint(packed_enc_out, packed_pred_out)

        with autocast(enabled=False):
            if self.isfused:
                nll = RNNTFusedLoss(
                    joiner_out.float(), squeezed_targets,
                    enc_out_lens.to(device=joiner_out.device),
                    expd_target_lens.to(device=joiner_out.device))
            else:
                nll = RNNTLoss(
                    joiner_out.float(), squeezed_targets,
                    enc_out_lens.to(device=joiner_out.device),
                    expd_target_lens.to(device=joiner_out.device),
                    gather=True, compact=True)

        # nll: (bs*n_hyps, )
        nll = (-nll).exp()
        # den: (bs, )
        den = (nll * mask_batches).sum(dim=1)
        # den: (bs, ) -> (bs*n_hyps, )
        den = den[torch.repeat_interleave(cnt_hypos)]
        return torch.sum(nll / den * penalty) / bs

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        enc_out, enc_out_len = self.encoder(inputs, input_lengths)

        return self.werloss(
            enc_out,
            enc_out_len,
            targets,
            target_lengths
        )


def build_model(cfg: dict, args: argparse.Namespace, dist: bool = True) -> DiscTransducerTrainer:

    assert 'MWER' in cfg, f"missing 'MWER' in field:"
    assert 'decoder' in cfg['MWER'], f"missing 'decoder' in field:MWER:"
    cfg['MWER']['trainer'] = cfg['MWER'].get('trainer', {})

    encoder, predictor, joiner = rnnt_builder(cfg, dist=False, wrapped=False)

    rnnt_decoder = RNNTDecoder(
        decoder=predictor,
        joint=joiner,
        **cfg['MWER']['decoder']
    )
    model = DiscTransducerTrainer(
        rnnt_decoder,
        encoder,
        predictor,
        joiner,
        **cfg['MWER']['trainer'])

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
