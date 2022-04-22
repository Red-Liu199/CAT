# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# MWER training of RNN-T

from ..shared import Manager
from ..shared import coreutils
from ..shared.decoder import AbsDecoder
from ..shared.encoder import AbsEncoder
from ..shared.manager import train as default_train_func
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateTransducer
)
from .beam_search import BeamSearcher as RNNTDecoder
from .train import build_model as rnnt_builder
from .train import TransducerTrainer
from .joint import (
    PackedSequence,
    AbsJointNet
)

from warp_rnnt import rnnt_loss as RNNTLoss
from warp_rnnt import fused_rnnt_loss_ as RNNTFusedLoss

import os
import jiwer
import math
import gather
import argparse
from collections import OrderedDict
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


def cal_wer(gt: Union[str, List[int]], hy: Union[str, List[int]]):
    return jiwer.compute_measures(gt, hy)['wer']


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
            batched_hypos = self.searcher.batching_rna(encoder_out, frame_lens)
            self.joint.requires_grad_(True)
            self.decoder.requires_grad_(True)

        frame_lens = frame_lens.to(torch.int)
        target_lens = target_lens.to(torch.int)
        loss = 0.
        for n in range(len(batched_hypos)):
            n_hyps = len(batched_hypos[n])
            # penalty: (n_hyps, )
            penalty = encoder_out.new_tensor(
                [
                    cal_wer(gt, hy)
                    for gt, hy in zip(
                        [
                            ' '.join(str(x) for x in targets[n, :target_lens[n]].cpu(
                            ).tolist())
                        ]*n_hyps,
                        [
                            ' '.join(str(x) for x in hyps.pred[1:])
                            for hyps in batched_hypos[n]
                        ]
                    )
                ]
            )

            if penalty[0] == 0.:
                batched_hypos[n] = batched_hypos[n][1:]
                penalty = penalty[1:]
                n_hyps -= 1
            if n_hyps == 0:
                continue
            # (n_hyps, U_n_max)
            cur_pred_in = coreutils.pad_list(
                [
                    targets.new_tensor(hypo.pred)
                    for hypo in batched_hypos[n]
                ]
            )
            cur_targets = targets.new_tensor(
                sum([hypo.pred[1:] for hypo in batched_hypos[n]], tuple())
            )
            # (n_hyps, )
            cur_target_lens = target_lens.new_tensor(
                [len(hypo)-1 for hypo in batched_hypos[n]])

            # (n_hyps, U_n_max)
            preditor_out, _ = self.decoder(
                cur_pred_in, input_lengths=cur_target_lens+1)

            # (n_hyps, T_n, V)
            cur_enc_out = encoder_out[n:n+1,
                                      :frame_lens[n], :].repeat(n_hyps, 1, 1)
            # (n_hyps, )
            cur_enc_len = frame_lens[n:n+1].repeat(n_hyps)
            packed_enc_out = PackedSequence()
            packed_enc_out._data = cur_enc_out.view(-1, cur_enc_out.size(-1))
            packed_enc_out._lens = cur_enc_len
            packed_pred_out = PackedSequence(preditor_out, cur_target_lens+1)
            if self.isfused:
                joiner_out = self.joint.impl_forward(
                    packed_enc_out, packed_pred_out)
            else:
                joiner_out = self.joint(packed_enc_out, packed_pred_out)

            with autocast(enabled=False):
                if self.isfused:
                    nll = RNNTFusedLoss(
                        joiner_out.float(), cur_targets.to(dtype=torch.int32),
                        cur_enc_len.to(device=joiner_out.device,
                                       dtype=torch.int32),
                        cur_target_lens.to(device=joiner_out.device, dtype=torch.int32))
                else:
                    nll = RNNTLoss(
                        joiner_out.float(), cur_targets.to(dtype=torch.int32),
                        cur_enc_len.to(device=joiner_out.device,
                                       dtype=torch.int32),
                        cur_target_lens.to(
                            device=joiner_out.device, dtype=torch.int32),
                        gather=True, compact=True)

            loss += torch.sum((-nll).exp()*penalty)

        return loss/encoder_out.size(0)

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
