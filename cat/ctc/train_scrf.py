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
from ..shared.manager import train as default_train_func
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
        func_eval=custom_evaluate,
        func_train=custom_train
    )


def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(
        dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def score_prior(model: nn.Module, in_tokens: torch.Tensor, targets: torch.Tensor, input_lengths: torch.LongTensor, do_normalize: bool = True):
    """Compute the prior score of the given model and input.

    This is almost the same as cat.shared.decoder.AbsDecoder.score,
    with the support to skipping log_softmax normalization.
    """
    U = input_lengths.max()
    if in_tokens.size(1) > U:
        in_tokens = in_tokens[:, :U]
    if targets.size(1) > U:
        targets = targets[:, :U]

    # logits: (N, U, K)
    logits, _ = model(in_tokens, input_lengths=input_lengths)
    if do_normalize:
        logits = logits.log_softmax(dim=-1)
    # score: (N, U)
    score = logits.gather(
        index=targets.long().unsqueeze(2), dim=-1).squeeze(-1)
    # True for not masked, False for masked, (N, U)
    score *= torch.arange(in_tokens.size(1), device=in_tokens.device)[
        None, :] < input_lengths[:, None].to(in_tokens.device)
    # (N,)
    score = score.sum(dim=-1)
    return score


class SCRFTrainer(AMTrainer):
    def __init__(
            self,
            lm: AbsDecoder,
            lm_weight: float = 1.0,
            num_aux_weight: float = 0.,
            n_samples: int = 256,
            local_normalized_encoder: bool = True,
            tuning_prior: bool = False,
            local_normalized_prior: bool = True,
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
        if tuning_prior:
            self._prior = lm
            lm.requires_grad_(True)
        else:
            self._prior = None

        self.weights = {
            'lm_weight': lm_weight,
            'num_weight': num_aux_weight
        }

        self.n_samples = n_samples
        self.local_normalized_encoder = local_normalized_encoder
        self.local_normalized_prior = local_normalized_prior
        self.criterion = nn.CTCLoss(reduction='none', zero_infinity=True)
        self._pad = nn.ConstantPad1d((1, 0), 0)

    def _IS(self, logits: torch.Tensor, lx: torch.Tensor):
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
        N, T, V = logits.shape
        K = self.n_samples

        # (K, N, T) -> (N, T, K)
        m = torch.distributions.categorical.Categorical(logits=logits)
        pi_samples = m.sample((K, )).permute(1, 2, 0)

        # (N, T, K) -> (N, K, T) -> (N*K, T)
        ysamples, lsamples = ctc_align.align_(
            pi_samples.transpose(1, 2).contiguous().view(-1, T),
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1)
        )
        ysamples = ysamples[:, :lsamples.max()]
        ysamples *= torch.arange(ysamples.size(1), device=ysamples.device)[
            None, :] < lsamples[:, None]

        # (UN, )
        unique_samples, inverse, ordered = unique(ysamples, dim=0)

        padded_ys = self._pad(unique_samples)
        # <s> A B C -> A B C <s>
        dummy_targets = torch.roll(padded_ys, -1, dims=1)

        # piror(y): (UN, )
        # here indeed is the log prob.
        prior_ys = score_prior(
            self.attach['lm'],
            padded_ys,
            dummy_targets,
            lsamples[ordered]+1,
            self.local_normalized_prior
        ) * self.weights['lm_weight']
        # (UN, ) -> (N*K, ) -> (N, K)
        prior_ys = torch.gather(prior_ys, dim=0, index=inverse).view(N, K)
        w_hat = torch.log_softmax(prior_ys.detach(), dim=1)
        return w_hat.exp(), prior_ys, pi_samples, lsamples, unique_samples, inverse, ordered

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):

        device = feats.device
        logits, lx = self.am(feats, lx)
        lx = lx.to(device=device, dtype=torch.int)
        ly = ly.to(device=device, dtype=torch.int)
        labels = labels.to(torch.int)

        N, T, V = logits.shape
        K = self.n_samples

        w_hat, prior_ys, pi_samples, lsamples, unique_samples, inverse, ordered = self._IS(
            logits, lx)
        squeeze_ratio = 1 - ordered.size(0) / N / K

        if self.local_normalized_encoder:
            score = logits.log_softmax(dim=-1)
        else:
            score = logits.float()

        if self._compute_y_grad:
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

        s += prior_ys
        # (N, ), i.e. the mu_hat = \sum (w_hat_i * s_i)
        den = torch.sum(s * w_hat, dim=1)

        # numerator: (N, )
        num = self.criterion(
            score.transpose(0, 1),
            labels.to(device='cpu'),
            lx, ly
        )
        if self._prior:
            # tunable prior network
            padded_ys = self._pad(coreutils.pad_list(
                torch.split(labels, ly.cpu().tolist())))
            # <s> A B C -> A B C <s>
            dummy_targets = torch.roll(padded_ys, -1, dims=1)

            # here indeed is the log prob.
            gtscore = score_prior(
                self.attach['lm'],
                padded_ys,
                dummy_targets,
                ly+1,
                self.local_normalized_prior
            ) * self.weights['lm_weight']
            num -= gtscore
            
            ########## DEBUG CODE ###########
            # print(torch.sum(w_hat*prior_ys, dim=1))
            # print(gtscore)
            # raise StopIteration
            #################################
            

        return ((1+self.weights['num_weight'])*num + den).mean(dim=0), squeeze_ratio, num


def custom_hook(manager, model, args, n_step, nnforward_args):

    loss, squeeze_ratio, ctc_loss = model(*nnforward_args)

    fold = args.grad_accum_fold
    if args.rank == 0 and n_step % fold == 0:
        manager.writer.add_scalar(
            'sample/squeeze-ratio', squeeze_ratio, manager.step)
        manager.writer.add_scalar(
            'loss/numerator', ctc_loss.detach().mean(dim=0).item(), manager.step)

    return loss


def custom_train(*args):
    return default_train_func(*args, hook_func=custom_hook)


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
