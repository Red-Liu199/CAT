"""
RNN-T training with sampling-based CRF.
"""

from .train import (
    TransducerTrainer,
    build_model as build_base_model,
    main_worker as basic_worker
)
from .beam_search import BeamSearcher as RNNTDecoder
from ..lm import lm_builder
from ..shared import coreutils, layer
from ..shared.decoder import AbsDecoder
from ..shared.manager import train as default_train_func
from ..ctc.train import (
    cal_wer,
    custom_evaluate,
    build_beamdecoder
)
from ..ctc.train_scrf import (
    unique,
    score_prior
)

import random
import ctc_align
import argparse
from typing import *
from warp_rnnt import rnnt_loss_simple

import torch
import torch.nn as nn


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    basic_worker(
        gpu, ngpus_per_node, args,
        func_build_model=build_model,
        func_train=custom_train,
        func_eval=custom_evaluate
    )


class CTCHEAD(nn.Module):
    def __init__(self, hdim: int, num_classes: int) -> None:
        super().__init__()
        self.rnn = layer._LSTM(hdim, hdim)
        self.fc = nn.Linear(hdim, num_classes)

    def forward(self, x, lx):
        return self.fc(self.rnn(x, lx)[0])


class STRFTransducerTrainer(TransducerTrainer):
    def __init__(
            self,
            hdim_last_enc_hid: int,
            num_classes: int,
            lm: Optional[AbsDecoder] = None,
            lm_weight: float = 1.0,
            num_aux_weight: float = 0.,
            n_samples: int = 256,
            decoder=None,
            **kwargs):
        super().__init__(**kwargs)
        assert isinstance(n_samples, int) and n_samples > 0
        assert isinstance(lm_weight, (int, float)) and lm_weight > 0
        assert isinstance(num_aux_weight, (int, float))

        self.n_samples = n_samples
        self.attach = {
            'decoder': decoder,
            'lm': lm
        }
        self.weights = {
            'lm_weight': lm_weight,
            'num_weight': num_aux_weight
        }
        self._pad = nn.ConstantPad1d((1, 0), 0)
        self.ctc_loss = nn.CTCLoss(reduction='none')
        self.aux_ctc_head = CTCHEAD(hdim_last_enc_hid, num_classes)
        self.head = nn.Linear(hdim_last_enc_hid, num_classes)

    @torch.no_grad()
    def dget_wer(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens: torch.LongTensor) -> torch.FloatTensor:

        if '_beam_decoder' not in self.attach:
            # register beam searcher
            self.attach['_beam_decoder'] = RNNTDecoder(
                self.predictor,
                self.joiner,
                beam_size=4,
                lm_module=self.attach['lm'],
                alpha=self.weights['lm_weight']
            )
        enc_out, lx = self.encoder(inputs, in_lens)
        lx = lx.to(torch.int32)
        bs = enc_out.size(0)
        batched_hypos = self.attach['_beam_decoder'].batch_decode(
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

    @torch.no_grad()
    def get_wer(self, xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):
        if self.attach['decoder'] is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: self.attach['decoder'] is not initialized.")

        bs = xs.size(0)
        last_enc_hid, lx = self.encoder(xs, lx)
        logits = self.aux_ctc_head(last_enc_hid, lx)

        # y_samples: (N, k, L), ly_samples: (N, k)
        y_samples, _, _, ly_samples = self.attach['decoder'].decode(
            logits.float().cpu(), lx.cpu())

        ground_truth = [ys[i, :ly[i]] for i in range(ys.size(0))]
        hypos = [y_samples[n, 0, :ly_samples[n, 0]].tolist()
                 for n in range(bs)]

        return cal_wer(ground_truth, hypos)

    @torch.no_grad()
    def _IS(self, ctc_out: torch.Tensor, lx: torch.Tensor):
        """modified from cat.ctc.train_scrf

        ctc_out : (T, N, V), log probs
        """

        T = ctc_out.size(0)
        K = self.n_samples

        # (T, N, V) -> (K, T, N)
        m = torch.distributions.categorical.Categorical(ctc_out.exp())
        # (K, T, N) -> (N, K, T)
        pi_samples = m.sample((K, )).permute(2, 0, 1).contiguous()

        # (N, K, T) -> (N*K, T)
        ysamples, lsamples = ctc_align.align_(
            pi_samples.view(-1, T),
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1)
        )
        # FIXME: a hack to avoid lsamples == 1
        lsamples[lsamples == 0] = 1
        ysamples[lsamples == 0] = random.randint(0, self.head.out_features-1)

        # (N*K, T) -> (N*K, U)
        ysamples = ysamples[:, :lsamples.max()]
        ysamples *= torch.arange(ysamples.size(1), device=ysamples.device)[
            None, :] < lsamples[:, None]

        # (N*K, U) -> (N', U)
        unique_samples, inverse, ordered = unique(ysamples, dim=0)

        return unique_samples, lsamples[ordered], inverse, ordered

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):

        device = feats.device
        last_enc_hid, lx = self.encoder(feats, lx)
        pred_out, _ = self.predictor(self._pad(labels))

        lx = lx.to(device=device, dtype=torch.int)
        ly = ly.to(device=device, dtype=torch.int)
        labels = labels.to(torch.int)

        N = last_enc_hid.size(0)
        K = self.n_samples

        # numerator: (N, )
        enc_out = self.head(last_enc_hid)

        num = rnnt_loss_simple(
            f_enc=enc_out,
            g_pred=pred_out,
            labels=labels,
            lf=lx,
            ll=ly
        )

        # (T, N, V)
        ctc_log_probs = self.aux_ctc_head(
            last_enc_hid.detach(), lx).log_softmax(dim=-1).transpose(0, 1)
        # joint ctc loss
        num_ctc = self.ctc_loss(ctc_log_probs, labels.to(
            device='cpu'), lx, ly).mean(dim=0)

        # denominator
        ## sample from ctc aux dist
        unique_samples, lsamples, inverse, ordered = self._IS(
            ctc_log_probs, lx)
        squeeze_ratio = 1 - ordered.size(0) / N / K

        ## piror(y): (N', )
        padded_ys = self._pad(unique_samples)
        if self.attach['lm'] is None:
            prior_ys = 0.
        else:
            ## <s> A B C -> A B C <s>
            dummy_targets = torch.roll(padded_ys, -1, dims=1)
            prior_ys = score_prior(
                self.attach['lm'],
                padded_ys,
                dummy_targets,
                lsamples+1
            ) * self.weights['lm_weight']

        ordered_src = torch.div(ordered, K, rounding_mode='floor')
        expand_lx = lx[ordered_src]

        ## calculate log P_ctc (N', )
        score_ctc = -self.ctc_loss(
            ctc_log_probs.detach().transpose(
                0, 1)[ordered_src].transpose(0, 1),
            unique_samples.to(device='cpu'),
            expand_lx, lsamples
        )

        ## calculate rnnt score (N', )
        score_rnnt = -rnnt_loss_simple(
            f_enc=enc_out[ordered_src].contiguous(),
            g_pred=self.predictor(padded_ys)[0].contiguous(),
            labels=unique_samples,
            lf=expand_lx,
            ll=lsamples,
            reduction='none'
        )

        ## w_hat = log(w_i): (N', ) -> (N, K)
        w_hat = torch.gather(
            score_rnnt.detach() + prior_ys -
            score_ctc, dim=0, index=inverse
        ).view(N, K)

        score_rnnt = torch.gather(
            score_rnnt, dim=0, index=inverse).view(N, K)

        den = (torch.softmax(w_hat, dim=1) * score_rnnt).sum(dim=1)

        return (1+self.weights['num_weight'])*num + den.mean(dim=0) + num_ctc, squeeze_ratio, num, num_ctc


def custom_hook(manager, model, args, n_step, nnforward_args):

    loss, squeeze_ratio, numerator_loss, aux_ctc_loss = model(*nnforward_args)

    fold = args.grad_accum_fold
    if args.rank == 0 and n_step % fold == 0:
        manager.writer.add_scalar(
            'sample/squeeze-ratio', squeeze_ratio, manager.step)
        manager.writer.add_scalar(
            'loss/numerator', float(numerator_loss), manager.step)
        manager.writer.add_scalar(
            'loss/aux-ctc-loss', float(aux_ctc_loss), manager.step)
    return loss


def custom_train(*args):
    return default_train_func(*args, hook_func=custom_hook)


def build_model(
        cfg: dict,
        args: Optional[Union[argparse.Namespace, dict]] = None,
        dist: bool = True,
        wrapped: bool = True):

    if args is not None:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        elif not isinstance(args, dict):
            raise ValueError(f"unsupport type of args: {type(args)}")

    assert 'trainer' in cfg, f"missing 'trainer' in field:"
    cfg_trainer = cfg['trainer']
    assert 'decoder' in cfg_trainer
    cfg_trainer['decoder'] = build_beamdecoder(cfg_trainer['decoder'])

    encoder, predictor, joiner = build_base_model(
        cfg, args, dist=False, wrapped=False)
    if not wrapped:
        return encoder, predictor, joiner

    cfg_trainer['encoder'] = encoder
    cfg_trainer['predictor'] = predictor
    cfg_trainer['joiner'] = joiner

    # initialize external lm
    if 'lm' in cfg_trainer:
        dummy_lm = lm_builder(coreutils.readjson(
            cfg_trainer['lm']['config']), dist=False)
        if cfg_trainer['lm'].get('check', None) is not None:
            coreutils.load_checkpoint(dummy_lm, cfg_trainer['lm']['check'])
        elm = dummy_lm.lm
        elm.eval()
        elm.requires_grad_(False)
        del dummy_lm
    else:
        elm = None

    cfg_trainer['lm'] = elm
    model = STRFTransducerTrainer(**cfg_trainer)

    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)
    if elm:
        elm.cuda(args['gpu'])
    model.cuda(args['gpu'])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args['gpu']])
    return model


def _parser():
    return coreutils.basic_trainer_parser("Sampling-base RNN-T-CRF Trainer")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    main()
