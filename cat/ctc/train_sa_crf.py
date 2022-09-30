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
import gather
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


def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(
        dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


class SCRFTrainer(AMTrainer):
    def __init__(
            self,
            lm: AbsDecoder,
            lm_weight: float = 1.0,
            num_aux_weight: float = 0.,
            n_samples: int = 256,
            local_normalized: bool = True,
            num_aux_weight_decay: float = 1.0,
            # compute gradients via mapped y seqs by CTC loss, instead of computing probs of pi seqs.
            compute_gradient_via_y: bool = False,
            **kwargs):
        super().__init__(**kwargs)
        assert isinstance(n_samples, int) and n_samples > 0
        assert isinstance(lm_weight, (int, float)) and lm_weight > 0
        assert isinstance(num_aux_weight, (int, float))
        assert not self.is_crf, f"sa-crf mode is conflict with crf"

        self._compute_y_grad = compute_gradient_via_y
        self.attach['lm'] = lm
        self.weights = {
            'lm_weight': lm_weight
        }
        # aux_ctc[t] = aux_ctc[t-1] * aux_ctc_decay
        if not local_normalized and num_aux_weight != 0.0:
            print(
                "warning: with a global normalized model, auxilliary numerator weight might not be proper.")

        # register as a buffer, so we can restore it when resuming from a stop training.
        self.register_buffer('_aux_ctc', torch.tensor(num_aux_weight))
        self._aux_decay_factor = num_aux_weight_decay

        self.n_samples = n_samples
        self._is_local_normalized = local_normalized
        self.criterion = nn.CTCLoss(reduction='none', zero_infinity=True)
        self._pad = nn.ConstantPad1d((1, 0), 0)

    def _IS(self, log_probs: torch.Tensor, lx: torch.Tensor):
        """Importance Sampling

        Return:
            (w_hat, pi_samples, y_samples, lysample)
            w_hat: (N, K)
            pi_samples: (N, T, K)
            y_samples: (N*K, U)
            lysample: (N*K, ), max(lysample) = U

        Note that in following formulas, p() is NOT prob().
        Use q(pi|x) = \prod_t prob(pi_t|x) as the proposal distribution.
        The importance weihgt w = p(pi|x) / q(pi|x) = prior(y) / Z(x),
            where the Z(x) is the normalized constant, we expect to avoid
            computing it, so we just return the re-normalized weight
            w_hat = [..., w_hat_i, ...], w_hat_i = prior(y_i) / \sum_j prior(y_j)
            here the piror(y) is generally P(Y)^lm_weight, P(Y) is obtained by a LM.
            If you're confused what this means, you may need to take a look
            at the Monte Carlo Sampling (expecially Importance Sampling.)
        """
        N, T, V = log_probs.shape
        K = self.n_samples

        # (N, T, K)
        pi_samples = torch.multinomial(
            log_probs.exp().view(-1, V),
            K, replacement=True
        ).view(N, T, K)

        # get alignment seqs
        # (N, T, K) -> (N, K, T) -> (N*K, T)
        ysamples = pi_samples.transpose(1, 2).contiguous().view(-1, T)

        # (N*K, T) -> (N*K, U)
        ysamples, lsamples = ctc_align.align_(
            ysamples,
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1)
        )
        ysamples = ysamples[:, :lsamples.max()]
        ysamples *= torch.arange(ysamples.size(1), device=ysamples.device)[
            None, :] < lsamples[:, None]

        padded_ys = self._pad(ysamples)
        # <s> A B C -> A B C <s>
        dummy_targets = torch.roll(padded_ys, -1, dims=1)

        # piror(y): (N*K, ) -> (N, K)
        # here indeed is the log prob.
        w_hat = self.attach['lm'].score(
            padded_ys,
            dummy_targets,
            lsamples+1
        ).view(N, K) * self.weights['lm_weight']
        w_hat = w_hat - torch.logsumexp(w_hat, dim=1, keepdim=True)
        return w_hat.exp(), pi_samples, ysamples, lsamples

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):

        device = feats.device
        logits, lx = self.am(feats, lx)
        log_probs = logits.log_softmax(dim=-1)
        lx = lx.to(device=device, dtype=torch.int)
        ly = ly.to(device=device, dtype=torch.int)
        labels = labels.to(torch.int)

        N, T, V = logits.shape
        K = self.n_samples

        w_hat, pi_samples, ysamples, lsamples = self._IS(log_probs, lx)

        if self._is_local_normalized:
            score = log_probs
        else:
            score = logits
        if self._compute_y_grad:
            unique_samples, inverse, ordered = unique(ysamples, dim=0)
            # (UN, )
            s = -self.criterion(
                # (N, T, V) -> (UN, T, V) -> (T, UN, V)
                score[torch.div(ordered, K, rounding_mode='floor')
                      ].transpose(0, 1),
                gather.cat(unique_samples, lsamples[ordered]).to(device='cpu'),
                # (N, ) -> (UN, )
                lx[torch.div(ordered, K, rounding_mode='floor')],
                lsamples[ordered]
            )
            # (UN, ) -> (N*K, )
            s = torch.gather(s, dim=0, index=inverse).view(N, K)
        else:
            # score(pi|X): (N, T, K)
            s = torch.gather(score, dim=-1, index=pi_samples)
            s *= (torch.arange(s.size(1), device=device)[
                None, :] < lx[:, None])[:, :, None]
            # (N, T, K) -> (N, K)
            s = s.sum(dim=1)

        # (N, ), a more numerical stable ver.
        estimate = torch.sum(s * w_hat, dim=1)

        # numerator: (N, )
        num = self.criterion(
            score.transpose(0, 1),
            labels.to(device='cpu'),
            lx, ly
        )
        if self.training:
            self._aux_ctc *= self._aux_decay_factor
        return ((1+self._aux_ctc)*num + estimate).mean(dim=0)


def build_model(cfg: dict, args: argparse.Namespace) -> Union[AbsEncoder, SCRFTrainer]:
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
    trainer_cfg['decoder']['alpha'] = trainer_cfg['decoder'].get(
        'alpha', trainer_cfg['lm'].get('weight', 1.0))
    if 'kenlm' not in trainer_cfg['decoder']:
        lmconfig = coreutils.readjson(trainer_cfg['lm']['config'])
        assert lmconfig['decoder']['type'] == 'NGram', \
            f"You do not set field:trainer:decoder:kenlm and field:trainer:lm:config is not directed to a kenlm."
        trainer_cfg['decoder']['kenlm'] = lmconfig['decoder']['kwargs']['f_binlm']
    trainer_cfg['lm'] = elm

    trainer_cfg['decoder'] = build_beamdecoder(
        trainer_cfg['decoder']
    )
    trainer_cfg['am'] = ctc_builder(cfg, args, dist=False, wrapper=False)

    model = SCRFTrainer(**trainer_cfg)

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
