# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# NCE training of RNN-T

__all__ = ["NCETransducerTrainer", "build_model", "_parser", "main"]

from ..shared.encoder import AbsEncoder
from ..shared.decoder import AbsDecoder, ILM
from ..shared.manager import train as default_train_func
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)
from ..shared import (
    Manager,
    coreutils
)
from ..lm import lm_builder
from .train import build_model as rnnt_builder
from .train import TransducerTrainer

from .joiner import AbsJointNet
from .beam_search import BeamSearcher as RNNTDecoder


import os
import jiwer
import math
import argparse
from typing import *
from tqdm import tqdm
from warp_rnnt import rnnt_loss as RNNTLoss

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
        args, build_model,
        func_train=custom_train,
        func_eval=custom_evaluate,
        extra_tracks=[
            'loss/nce-data',
            'loss/nce-noise',
            'acc/data',
            'acc/noise'
        ]
    )

    # training
    manager.run(args)


class NCETransducerTrainer(TransducerTrainer):
    def __init__(
            self,
            encoder: AbsEncoder,
            predictor: AbsDecoder,
            joiner: AbsJointNet,
            ext_lm: AbsDecoder,
            beamdecoder: RNNTDecoder,
            ilm_weight: float,
            elm_weight: float,
            mle_weight: Optional[float] = 0.,
            *args, **kwargs) -> None:
        super().__init__(encoder, predictor, joiner, *args, **kwargs)

        assert isinstance(ext_lm, AbsDecoder)
        assert isinstance(beamdecoder, RNNTDecoder)
        assert self._compact
        assert self.ilme_weight == 0.
        assert not self._sampled_softmax
        assert isinstance(mle_weight, float)
        assert isinstance(ilm_weight, float)
        assert isinstance(elm_weight, float)
        if mle_weight != 0.:
            raise NotImplementedError
        if ilm_weight > 0:
            print(
                f"warning: given ilm weight={ilm_weight} > 0, but commonly it's a negative one.")

        # store extra modules in a dict, so that state_dict() won't return their params.
        self.attach = {
            'elm': ext_lm,
            'searcher': beamdecoder,
            '_eval_beam_searcher': RNNTDecoder(
                predictor,
                joiner,
                beam_size=beamdecoder.beam_size,
                lm_module=ext_lm,
                alpha=elm_weight,
                est_ilm=(ilm_weight != 0.0),
                ilm_weight=ilm_weight
            )
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

            with autocast(enabled=False):
                # q_y_x: (N, )
                q_y_x = -RNNTLoss(
                    noise_join_out.float(), squeeze_targets,
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
        ilm_scores = ilm_log_prob.sum(dim=-1)

        # elm_scores: (N, )
        elm_scores = self.attach['elm'].score(
            padded_targets, dummy_targets, ly)

        # cal model score
        model_join_out = self.joiner(enc_out, pred_out, lx, ly+1)
        with autocast(enabled=False):
            # model_log_prob: (N, )
            model_log_prob = -RNNTLoss(
                model_join_out.float(), squeeze_targets,
                lx.to(device=device, dtype=torch.int32),
                ly.to(device=device, dtype=torch.int32),
                gather=True, compact=True
            )

        p_hat_y_x = model_log_prob + \
            self.weights['ilm'] * ilm_scores + \
            self.weights['elm'] * elm_scores

        return p_hat_y_x, q_y_x

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens: torch.LongTensor) -> torch.FloatTensor:

        enc_out, lx = self.encoder(inputs, in_lens)
        lx = lx.to(torch.int32)
        bs = enc_out.size(0)

        p_data, q_data = self.cal_g_bc(enc_out, targets, lx, target_lens)

        # draw noise samples
        with torch.no_grad():
            batched_hypos = self.attach['searcher'].batching_rna(enc_out, lx)

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

        p_noise, q_noise = self.cal_g_bc(
            noise_enc_out, noise_targets, noise_lx, noise_ly)

        noise_ratio = noise_enc_out.size(0) / bs

        nce_data_obj = self._logsigmoid(
            p_data - q_data - math.log(noise_ratio))
        nce_noise_obj = self._logsigmoid(
            q_noise + math.log(noise_ratio) - p_noise)

        return -(nce_data_obj.mean(dim=0) + noise_ratio * nce_noise_obj.mean(dim=0)), bs, \
            (nce_data_obj, nce_noise_obj)

    @torch.no_grad()
    def get_wer(self, inputs: torch.FloatTensor, targets: torch.LongTensor, in_lens: torch.LongTensor, target_lens: torch.LongTensor) -> torch.FloatTensor:

        enc_out, lx = self.encoder(inputs, in_lens)
        lx = lx.to(torch.int32)
        bs = enc_out.size(0)
        batched_hypos = self.attach['_eval_beam_searcher'].batching_rna(
            enc_out, lx)

        ground_truth = [
            ' '.join(
                str(x)
                for x in targets[n, :target_lens[n]].cpu().tolist()
            )
            for n in range(bs)
        ]
        hypos = [
            ' '.join(str(x) for x in list_hypos[0].pred[1:])
            for list_hypos in batched_hypos
        ]
        assert len(hypos) == len(ground_truth)

        err = cal_wer(ground_truth, hypos)
        cnt_err = sum(x for x, _ in err)
        cnt_sum = sum(x for _, x in err)
        return cnt_err, cnt_sum


def custom_hook(
        manager: Manager,
        model: NCETransducerTrainer,
        args: argparse.Namespace,
        n_step: int,
        nnforward_args: tuple):

    nce_loss, _, (p_c_0, p_c_1) = model(*nnforward_args)

    if args.rank == 0:
        # FIXME: not exact global accuracy
        pos_acc = ((p_c_0.exp() > 0.5).sum() / p_c_0.size(0)).item()
        noise_acc = ((p_c_1.exp() > 0.5).sum() / p_c_1.size(0)).item()
        loss_data = (-p_c_0.mean(dim=0)).item()
        loss_noise = (nce_loss - loss_data).item()
        step_cur = manager.step_by_last_epoch + n_step
        manager.monitor.update(
            {
                'loss/nce-data': (loss_data, step_cur),
                'loss/nce-noise': (loss_noise, step_cur),
                'acc/data': (pos_acc, step_cur),
                'acc/noise': (noise_acc, step_cur)
            }
        )
        manager.writer.add_scalar(
            'loss/nce-data', loss_data, step_cur)
        manager.writer.add_scalar(
            'loss/nce-noise', loss_noise, step_cur)
        manager.writer.add_scalar(
            'acc/data', pos_acc, step_cur)
        manager.writer.add_scalar(
            'acc/noise', noise_acc, step_cur)
    return nce_loss


def custom_train(*args):
    return default_train_func(*args, hook_func=custom_hook)


def cal_wer(gt: List[str], hy: List[str]) -> List[Tuple[int, int]]:
    def _get_wer(_gt, _hy):
        measure = jiwer.compute_measures(_gt, _hy)
        cnt_err = measure['substitutions'] + \
            measure['deletions'] + measure['insertions']
        cnt_sum = measure['substitutions'] + \
            measure['deletions'] + measure['hits']
        return (cnt_err, cnt_sum)
    return [_get_wer(_gt, _hy) for _gt, _hy in zip(gt, hy)]


@torch.no_grad()
def custom_evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:

    model = manager.model       # type: NCETransducerTrainer
    cnt_tokens = 0
    cnt_err = 0
    n_proc = dist.get_world_size()

    for i, minibatch in tqdm(enumerate(testloader), desc=f'Epoch: {manager.epoch} | eval',
                             unit='batch', total=len(testloader), disable=(args.gpu != 0), leave=False):

        feats, ilens, labels, olens = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)

        '''
        Suppose the loss is reduced by mean
        '''
        part_cnt_err, part_cnt_sum = model.module.get_wer(
            feats, labels, ilens, olens)

        gather_obj = [None for _ in range(n_proc)]
        dist.gather_object(
            (part_cnt_err, part_cnt_sum),
            gather_obj if args.rank == 0 else None,
            dst=0
        )
        if args.rank == 0:
            l_err, l_sum = list(zip(*gather_obj))
            cnt_err += sum(l_err)
            cnt_tokens += sum(l_sum)

    if args.rank == 0:
        wer = cnt_err / cnt_tokens
        manager.writer.add_scalar('loss/dev-wer', wer, manager.step)
        manager.monitor.update('eval:loss', (wer, manager.step))

        scatter_list = [wer]
    else:
        scatter_list = [None]

    dist.broadcast_object_list(scatter_list, src=0)
    return scatter_list[0]


def build_model(cfg: dict, args: argparse.Namespace, dist: bool = True) -> NCETransducerTrainer:
    """
    cfg:
        nce:
            init-n-model: # the noise model shares the encoder of training one.
                check:
            init-elm:
                config:
                check:
            decoder:
                ...
            trainer:
                ...
        # basic transducer config
        ...

    """
    assert 'nce' in cfg, f"missing 'nce' in field:"

    # initialize noise model.
    dummy_trainer = rnnt_builder(cfg, dist=False, wrapped=True)
    coreutils.load_checkpoint(
        dummy_trainer, cfg['nce']['init-n-model']['check'])
    dummy_trainer.eval()
    dummy_trainer.requires_grad_(False)
    beam_searcher = RNNTDecoder(
        predictor=dummy_trainer.predictor,
        joiner=dummy_trainer.joiner,
        **cfg['nce']['decoder']
    )
    del dummy_trainer
    beam_searcher.joiner.cuda(args.gpu)
    beam_searcher.predictor.cuda(args.gpu)

    # initialize external lm
    dummy_lm = lm_builder(coreutils.readjson(
        cfg['nce']['init-elm']['config']), dist=False)
    coreutils.load_checkpoint(dummy_lm, cfg['nce']['init-elm']['check'])
    elm = dummy_lm.lm
    elm.eval()
    elm.requires_grad_(False)
    elm.cuda(args.gpu)
    del dummy_lm

    # initialize the real model
    enc, pred, join = rnnt_builder(cfg, dist=False, wrapped=False)
    model = NCETransducerTrainer(
        encoder=enc,
        predictor=pred,
        joiner=join,
        ext_lm=elm,
        beamdecoder=beam_searcher,
        **cfg['nce']['trainer'],
        **cfg['transducer']
    )

    if not dist:
        return model

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
