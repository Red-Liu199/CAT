# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# NCE training of RNN-T

__all__ = ["NCETransducerTrainer", "build_model", "_parser", "main"]

from .train import (
    TransducerTrainer,
    build_model as rnnt_builder,
    main_worker as basic_worker
)
from .beam_search import BeamSearcher as RNNTDecoder
from ..ctc.train import cal_wer, custom_evaluate
from ..lm import lm_builder
from ..shared import coreutils
from ..shared.decoder import AbsDecoder, ILM
from ..shared.manager import (
    Manager,
    train as default_train_func
)


import os
import math
import argparse
import Levenshtein
from typing import *
from warp_rnnt import rnnt_loss as RNNTLoss

import torch
import torch.nn as nn


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):

    return basic_worker(
        gpu, ngpus_per_node, args,
        func_build_model=build_model,
        func_train=custom_train,
        func_eval=custom_evaluate
    )


class NCETransducerTrainer(TransducerTrainer):
    def __init__(
            self,
            ext_lm: AbsDecoder,
            beamdecoder: RNNTDecoder,
            ilm_weight: float,
            elm_weight: float,
            mle_weight: Optional[float] = 0.,
            trainable_weight: bool = False,
            **kwargs) -> None:
        super().__init__(**kwargs)

        assert isinstance(ext_lm, AbsDecoder)
        assert isinstance(beamdecoder, RNNTDecoder)
        assert self._compact
        assert self.ilme_weight == 0.
        try:
            ilm_weight = float(ilm_weight)
            elm_weight = float(elm_weight)
            mle_weight = float(mle_weight)
        except ValueError as ve:
            print("weights in invalid format: ilm | elm | mle = "
                  f"{ilm_weight} | {elm_weight} | {mle_weight}")
            exit(1)

        if ilm_weight > 0:
            print(
                f"warning: given ilm weight={ilm_weight} > 0, but commonly it's a negative one.")

        # store extra modules in a dict, so that state_dict() won't return their params.
        self.attach = {
            'elm': ext_lm,
            'searcher': beamdecoder,
            '_eval_beam_searcher': None
        }
        ilm = ILM(lazy_init=True)
        ilm._stem = self.predictor
        ilm._head = self.joiner
        self.attach['ilm'] = ilm
        self.weights = {
            'ilm': ilm_weight,   # weight for ILM
            'elm': elm_weight,   # weight for ELM
            'mle': mle_weight    # RNN-T loss weight for joint training
        }
        if trainable_weight:
            self._weight_elm = nn.parameter.Parameter(torch.tensor(elm_weight))
            self._weight_ilm = nn.parameter.Parameter(torch.tensor(ilm_weight))
            self.weights['elm'] = self._weight_elm
            self.weights['ilm'] = self._weight_ilm
        self.trainable_weight = trainable_weight

        self._pad = nn.ConstantPad1d((1, 0), 0)
        self._logsigmoid = nn.LogSigmoid()

    def cal_g_bc(
            self,
            enc_out: torch.Tensor,
            targets: torch.Tensor,
            lx: torch.Tensor,
            ly: torch.Tensor):
        """Calculate the 'g' variable for nce training."""
        assert enc_out.size(0) == targets.size(0)
        assert enc_out.size(0) == lx.size(0)
        assert enc_out.size(0) == ly.size(0)
        device = enc_out.device

        squeeze_targets = torch.cat([
            targets[n, :ly[n]]
            for n in range(ly.size(0))
        ], dim=0).to(device=device, dtype=torch.int32)

        padded_targets = self._pad(targets)
        # cal noise prob
        with torch.no_grad():
            noise_pred_out, _ = self.attach['searcher'].predictor(
                padded_targets, input_lengths=ly+1)
            noise_join_out = self.attach['searcher'].joiner(
                enc_out, noise_pred_out, lx, ly+1
            )

            # q_y_x: (N, )
            q_y_x = -RNNTLoss(
                noise_join_out, squeeze_targets,
                lx.to(device=device, dtype=torch.int32),
                ly.to(device=device, dtype=torch.int32),
                gather=True, compact=True
            )

        # cal ILM scores, refer to cat.shared.decoder.AbsDecoder.score()
        pred_out, _ = self.predictor(padded_targets, input_lengths=ly+1)
        # <s> A B C -> A B C <s>
        dummy_targets = torch.roll(padded_targets, -1, dims=1)
        # ilm_logit: (N, U, K)
        ilm_logit = self.joiner.forward_pred_only(pred_out, raw_logit=True)
        # normalize over non-blank labels
        ilm_logit[:, :, 0].fill_(torch.finfo(ilm_logit.dtype).min)
        # ilm_log_prob: (N, U)
        ilm_log_prob = ilm_logit.log_softmax(dim=-1).gather(
            index=dummy_targets.unsqueeze(2), dim=-1).squeeze(-1)

        # NOTE: be cautious that, there's no </s> in vocab,
        # ... so we only take U-1 labels into account for scores
        # ... following ly is originally (ly+1)
        # True for not masked, False for masked, (N, U)
        padding_mask = torch.arange(padded_targets.size(1), device=device)[
            None, :] < (ly)[:, None].to(device)
        ilm_log_prob *= padding_mask

        # ilm_scores: (N, U) -> (N, )
        if self.trainable_weight or self.weights['ilm'] != 0.:
            ilm_scores = ilm_log_prob.sum(dim=-1)
        else:
            ilm_scores = 0.

        # elm_scores: (N, )
        if self.trainable_weight or self.weights['elm'] != 0.:
            elm_scores = self.attach['elm'].score(
                padded_targets, dummy_targets, ly)
        else:
            elm_scores = 0.

        # cal model score
        model_join_out, squeeze_targets, lx, ly = self.compute_join(
            enc_out, pred_out, squeeze_targets, lx, ly
        )
        # p_y_x: (N, )
        p_y_x = -RNNTLoss(
            model_join_out,
            squeeze_targets, lx, ly,
            gather=True, compact=True
        )

        p_hat_y_x = p_y_x + \
            self.weights['ilm'] * ilm_scores + \
            self.weights['elm'] * elm_scores

        return p_hat_y_x, q_y_x, p_y_x

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens: torch.LongTensor) -> torch.FloatTensor:

        enc_out, lx = self.encoder(inputs, in_lens)
        lx = lx.to(torch.int32)
        bs = enc_out.size(0)

        p_data, q_data, p_raw_data = self.cal_g_bc(
            enc_out, targets, lx, target_lens)

        # draw noise samples
        with torch.no_grad():
            batched_hypos = self.attach['searcher'].batching_rna(enc_out, lx)
            for hypos in batched_hypos:
                for i in range(len(hypos)-1, -1, -1):
                    if len(hypos[i]) == 1:
                        hypos.pop(i)

            # get number of negative samples w.r.t. each utterances
            cnt_hypos = lx.new_tensor([len(list_hypos)
                                      for list_hypos in batched_hypos])
            batched_tokens = sum(
                ([hypo.pred for hypo in list_hypos]
                 for list_hypos in batched_hypos), [])   # type: List[Tuple[int]]
            del batched_hypos

        noise_enc_out = torch.cat([
            enc_out[n].expand(cnt_hypos[n:n+1], -1, -1)
            for n in range(bs)
        ], dim=0)
        noise_lx = torch.cat([
            lx[n:n+1].expand(cnt_hypos[n])
            for n in range(bs)
        ], dim=0)
        noise_targets = coreutils.pad_list([
            targets.new_tensor(hypo[1:])
            for hypo in batched_tokens
        ])
        noise_ly = target_lens.new_tensor(
            [len(hypo)-1 for hypo in batched_tokens])
        del batched_tokens

        p_noise, q_noise, _ = self.cal_g_bc(
            noise_enc_out, noise_targets, noise_lx, noise_ly)

        noise_ratio = noise_enc_out.size(0) / bs

        nce_data_obj = self._logsigmoid(
            p_data - q_data - math.log(noise_ratio))
        nce_noise_obj = self._logsigmoid(
            q_noise + math.log(noise_ratio) - p_noise)

        nce_data_loss = -nce_data_obj.mean(dim=0)
        nce_noise_loss = -noise_ratio*nce_noise_obj.mean(dim=0)
        ml_loss = -p_raw_data.mean(dim=0)
        return ml_loss*self.weights['mle'] + nce_data_loss + nce_noise_loss, \
            (nce_data_obj.detach(), nce_noise_obj.detach()), \
            (nce_data_loss.detach(), nce_noise_loss.detach()), \
            ml_loss.detach(), (self.weights['ilm'], self.weights['elm'])

    @torch.no_grad()
    def get_wer(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens: torch.LongTensor) -> torch.FloatTensor:

        if self.attach['_eval_beam_searcher'] is None:
            # register beam searcher
            self.attach['_eval_beam_searcher'] = RNNTDecoder(
                self.predictor,
                self.joiner,
                beam_size=self.attach['searcher'].beam_size,
                lm_module=self.attach['elm'],
                alpha=self.weights['elm'],
                est_ilm=True,
                ilm_weight=self.weights['ilm']
            )
        enc_out, lx = self.encoder(inputs, in_lens)
        lx = lx.to(torch.int32)
        bs = enc_out.size(0)
        batched_hypos = self.attach['_eval_beam_searcher'].batching_rna(
            enc_out, lx)

        ground_truth = [
            targets[n, :target_lens[n]].cpu().tolist()
            for n in range(bs)
        ]
        hypos = [
            list_hypos[0].pred[1:]
            for list_hypos in batched_hypos
        ]

        return cal_wer(ground_truth, hypos)


def custom_hook(
        manager: Manager,
        model: NCETransducerTrainer,
        args: argparse.Namespace,
        n_step: int,
        nnforward_args: tuple):

    loss, (p_c_0, p_c_1), (l_data, l_noise), l_ml, (ilm_w,
                                                    elm_w) = model(*nnforward_args)

    if args.rank == 0:
        # FIXME: not exact global accuracy
        l_data = l_data.item()
        l_noise = l_noise.item()
        l_ml = l_ml.item()
        pos_acc = ((p_c_0.exp() > 0.5).sum() / p_c_0.size(0)).item()
        noise_acc = ((p_c_1.exp() > 0.5).sum() / p_c_1.size(0)).item()
        step_cur = manager.step_by_last_epoch + n_step
        manager.writer.add_scalar(
            'loss/ml', l_ml, step_cur)
        manager.writer.add_scalar(
            'loss/nce', (l_data+l_noise), step_cur)
        manager.writer.add_scalar(
            'loss/nce-data', l_data, step_cur)
        manager.writer.add_scalar(
            'loss/nce-noise', l_noise, step_cur)
        manager.writer.add_scalar(
            'acc/data', pos_acc, step_cur)
        manager.writer.add_scalar(
            'acc/noise', noise_acc, step_cur)
        manager.writer.add_scalar(
            'weight/ilm', float(ilm_w), step_cur)
        manager.writer.add_scalar(
            'weight/elm', float(elm_w), step_cur)
    return loss


def custom_train(*args):
    return default_train_func(*args, hook_func=custom_hook)


def build_model(cfg: dict, args: argparse.Namespace, dist: bool = True) -> NCETransducerTrainer:
    """
    cfg:
        trainer:
            init-n-model: # the noise model shares the encoder of training one.
                check:
            init-elm:     # settings for the ELM
                config:
                check:
            decoder:      # settings for the beam search decoder 
                ...
            ...     # other settings
        ...

    """

    # initialize noise model.
    dummy_trainer = rnnt_builder(cfg, dist=False, wrapped=True)
    coreutils.load_checkpoint(
        dummy_trainer, cfg['trainer']['init-n-model']['check'])
    dummy_trainer.eval()
    dummy_trainer.requires_grad_(False)
    beam_searcher = RNNTDecoder(
        predictor=dummy_trainer.predictor,
        joiner=dummy_trainer.joiner,
        **cfg['trainer']['decoder']
    )
    del dummy_trainer

    # initialize external lm
    dummy_lm = lm_builder(coreutils.readjson(
        cfg['trainer']['init-elm']['config']), dist=False)
    if cfg['trainer']['init-elm'].get('check', None) is not None:
        coreutils.load_checkpoint(
            dummy_lm, cfg['trainer']['init-elm']['check'])
    elm = dummy_lm.lm
    elm.eval()
    elm.requires_grad_(False)
    del dummy_lm

    # initialize the real model
    enc, pred, join = rnnt_builder(cfg, dist=False, wrapped=False)
    cfg['trainer'].update({
        'encoder': enc,
        'predictor': pred,
        'joiner': join
    })

    model = NCETransducerTrainer(
        ext_lm=elm,
        beamdecoder=beam_searcher,
        **cfg['trainer']
    )

    if not dist:
        return model
    beam_searcher.joiner.cuda(args.gpu)
    beam_searcher.predictor.cuda(args.gpu)
    elm.cuda(args.gpu)

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


def _parser():
    return coreutils.basic_trainer_parser("NCE Transducer Training")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()