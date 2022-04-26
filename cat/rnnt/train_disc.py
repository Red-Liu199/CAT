# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)

from ..shared import Manager
from ..shared import coreutils
from ..shared.decoder import AbsDecoder
from ..shared.encoder import AbsEncoder
from ..shared.manager import train as default_train_func
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)
from ..ctc.train import build_model as ctc_builder
from .train import build_model as rnnt_builder
from .joiner import (
    PackedSequence,
    AbsJointNet
)
from ctcdecode import CTCBeamDecoder
import warp_rnnt

import os
import math
import gather
import argparse
from collections import OrderedDict

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
        func_train=train, extra_tracks=['pos acc', 'noise acc'])

    # training
    manager.run(args)


class DiscTransducerTrainer(nn.Module):
    def __init__(self,
                 encoder: AbsEncoder,
                 decoder: AbsDecoder,
                 joiner: AbsJointNet,
                 ctc_sampler: AbsEncoder,
                 searcher: CTCBeamDecoder,
                 beta: float = 0.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        self.sampler = ctc_sampler
        self.searcher = searcher
        self.ctc_loss = nn.CTCLoss(reduction='none')
        self._pad = nn.ConstantPad1d((1, 0), 0)
        # length normalization factor
        self._beta = beta
        self._logsigmoid = nn.LogSigmoid()

    def cal_p(self,
              model_xs: torch.Tensor, lx: torch.Tensor,
              noise_xs: torch.Tensor, lnx: torch.Tensor,
              ys: torch.Tensor, ly: torch.Tensor,
              nu: int, is_noise: bool = False):
        """
        model_xs : [Stu, V]
        lx: [N, ]
        noise_xs: [N, T, V]
        lnx: [N, ]
        ys: [Su, ]
        ly: [N, ]
        """
        assert model_xs.dim() == 2
        assert noise_xs.dim() == 3
        assert lx.size(0) == ly.size(0) and lx.size(
            0) == noise_xs.size(0) and lx.size(0) == lnx.size(0)

        # Q
        with torch.no_grad(), autocast(enabled=False):
            # [N, T, V] -> [T, N, V]
            noise_xs = noise_xs.transpose(0, 1)
            noise_log_probs = - \
                self.ctc_loss(noise_xs.float().log_softmax(
                    dim=-1), ys, lnx, ly)

        # P
        with autocast(enabled=False):
            ys = gather.cat(ys.unsqueeze(2).float(), ly).squeeze(0)
            dist_log_probs = -warp_rnnt.rnnt_loss(
                model_xs.float(), ys.to(torch.int), lx, ly,
                reduction='none', gather=True, compact=True) + self._beta * ly

        q_nu = math.log(nu) + noise_log_probs

        # G = log(P) - log(vQ) = dist_log_probs - q_nu
        if is_noise:
            return self._logsigmoid(q_nu - dist_log_probs)
        else:
            return self._logsigmoid(dist_log_probs - q_nu)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor):

        K = self.searcher._beam_width
        targets = targets.to(inputs.device, non_blocking=True)
        output_encoder, encoder_lens = self.encoder(inputs, input_lengths)
        encoder_lens = encoder_lens.to(torch.int)
        target_lengths = target_lengths.to(torch.int)

        # positive samples
        with torch.no_grad():
            sampler_out, sampler_lens = self.sampler(inputs, input_lengths)
            sampler_lens = sampler_lens.to(torch.int)
            padded_targets = self._pad(targets)
            pos_decoder_out, _ = self.decoder(
                padded_targets, input_lengths=target_lengths+1)
        pos_joinout = self.joiner(
            PackedSequence(output_encoder, encoder_lens),
            PackedSequence(pos_decoder_out, target_lengths+1)
        )

        pos_logp = self.cal_p(pos_joinout, encoder_lens, sampler_out,
                              sampler_lens, targets, target_lengths, K, False)
        # noise samples
        with torch.no_grad():
            # draw noise samples
            # [N, K, Umax]      [N, K]
            noise_samples, _, _, l_hypos = self.searcher.decode(
                sampler_out, seq_lens=sampler_lens)
            noise_samples = noise_samples[:, :, :l_hypos.max()]
            # [N, K, U] -> [K, N, U] -> [KN, U]
            noise_samples = noise_samples.transpose(
                0, 1).contiguous().view(-1, noise_samples.size(-1)).to(inputs.device, non_blocking=True)
            l_hypos = l_hypos.transpose(0, 1).contiguous(
            ).view(-1).to(inputs.device, non_blocking=True)

            # [NK, T, V]
            noise_sampler_out = sampler_out.repeat(K, 1, 1).contiguous()
            # [N, ] -> [NK, ]
            noise_sampler_lens = sampler_lens.repeat(K)

            padding_mask = torch.arange(noise_samples.size(1), device=noise_samples.device)[
                None, :] < l_hypos[:, None].to(noise_samples.device)
            noise_samples *= padding_mask
            noise_decoder_out, _ = self.decoder(
                self._pad(noise_samples), input_lengths=l_hypos+1)
        noise_encoder_out = output_encoder.repeat(K, 1, 1).contiguous()
        noise_encoder_lens = encoder_lens.repeat(K)
        noise_joinout = self.joiner(
            PackedSequence(noise_encoder_out, noise_encoder_lens),
            PackedSequence(noise_decoder_out, l_hypos+1))

        noise_logp = self.cal_p(
            noise_joinout, noise_encoder_lens,
            noise_sampler_out, noise_sampler_lens,
            noise_samples, l_hypos, K, True)

        return (-(pos_logp.mean() + K*noise_logp.mean()), inputs.size(0)), \
            pos_logp.detach().exp_() > 0.5, noise_logp.detach().exp_() > 0.5

    def train(self, mode: bool = True):
        super().train(mode)
        self.sampler.eval()
        return


def custom_hook(
        manager: Manager,
        model: DiscTransducerTrainer,
        args: argparse.Namespace,
        n_step: int,
        nnforward_args: tuple):

    loss, TP, FP = model(*nnforward_args)

    if args.rank == 0:
        # FIXME: not exact global accuracy
        pos_acc = (TP.sum()/TP.size(0)).item()
        noise_acc = (FP.sum()/FP.size(0)).item()
        manager.monitor.update('pos acc', pos_acc)
        manager.monitor.update('noise acc', noise_acc)
        manager.writer.add_scalar(
            'acc/positive', pos_acc, manager.step + n_step)
        manager.writer.add_scalar(
            'acc/noise', noise_acc, manager.step + n_step)
    return loss


def train(*args):
    return default_train_func(*args, _trainer_hook=custom_hook)


def build_model(args, cfg: dict, dist: bool = True) -> DiscTransducerTrainer:
    def _load_and_immigrate(orin_dict_path: str, str_src: str, str_dst: str) -> OrderedDict:
        if not os.path.isfile(orin_dict_path):
            raise FileNotFoundError(f"{orin_dict_path} is not a valid file.")

        checkpoint = torch.load(orin_dict_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_state_dict[k.replace(str_src, str_dst)] = v
        del checkpoint
        return new_state_dict

    if 'DiscTransducerTrainer' not in cfg:
        cfg['DiscTransducerTrainer'] = {}

    assert 'ctc-sampler' in cfg
    ctc_config = cfg['ctc-sampler']
    assert 'pretrain-config' in ctc_config
    assert 'pretrain-check' in ctc_config
    ctc_model = ctc_builder(
        coreutils.readjson(ctc_config['pretrain-config']),
        dist=False,
        wrapper=False
    )
    ctc_model.load_state_dict(_load_and_immigrate(
        ctc_config['pretrain-check'], 'module.am.', ''))
    ctc_model.eval()
    ctc_model.requires_grad_(False)

    assert 'searcher' in cfg
    encoder, decoder, joiner = rnnt_builder(
        cfg, args,  dist=False, wrapped=False)

    labels = [str(i) for i in range(ctc_model.classifier.out_features)]
    searcher = CTCBeamDecoder(
        labels, log_probs_input=True, **cfg['searcher'])

    model = DiscTransducerTrainer(
        encoder, decoder, joiner, ctc_model, searcher, **cfg['DiscTransducerTrainer'])

    if not dist:
        setattr(model, 'requires_slice', True)
        return model

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    setattr(model, 'requires_slice', True)

    return model


if __name__ == "__main__":
    parser = coreutils.basic_trainer_parser()
    args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)
