# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)

from ..shared import Manager
from ..shared import coreutils as utils
from ..shared.decoder import AbsDecoder
from ..shared.encoder import AbsEncoder
from .train import build_model as rnnt_builder
from ..ctc.train import build_model as ctc_builder
from .joint import (
    PackedSequence,
    AbsJointNet
)
from ..shared.data import (
    SpeechDatasetPickle,
    sortedPadCollateTransducer
)
from ctcdecode import CTCBeamDecoder
import warp_rnnt

import math
import os
import json
import gather
import argparse
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    manager = Manager(SpeechDatasetPickle,
                      sortedPadCollateTransducer(),
                      args, build_model,
                      func_train=train, extra_tracks=['Pos Acc', 'Noise Acc'])

    # training
    manager.run(args)


class DiscTransducerTrainer(nn.Module):
    def __init__(self,
                 encoder: AbsEncoder,
                 decoder: AbsDecoder,
                 joint: AbsJointNet,
                 ctc_sampler: AbsEncoder,
                 searcher: CTCBeamDecoder,
                 beta: float = 0.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint

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
        pos_joint_out = self.joint(
            PackedSequence(output_encoder, encoder_lens),
            PackedSequence(pos_decoder_out, target_lengths+1)
        )

        pos_logp = self.cal_p(pos_joint_out, encoder_lens, sampler_out,
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
        noise_joint_out = self.joint(
            PackedSequence(noise_encoder_out, noise_encoder_lens),
            PackedSequence(noise_decoder_out, l_hypos+1))

        noise_logp = self.cal_p(
            noise_joint_out, noise_encoder_lens,
            noise_sampler_out, noise_sampler_lens,
            noise_samples, l_hypos, K, True)

        return -(pos_logp.mean() + K*noise_logp.mean()), inputs.size(0), \
            pos_logp.detach().exp_() > 0.5, noise_logp.detach().exp_() > 0.5

    def train(self, mode: bool = True):
        super().train(mode)
        self.sampler.eval()
        return


def train(trainloader, args: argparse.Namespace, manager: Manager):

    def _go_step(detach_loss, n_batch):
        # we divide loss with fold since we want the gradients to be divided by fold
        with autocast(enabled=enableAMP):
            loss, norm_size, TP, FP = model(
                features, labels, input_lengths, label_lengths)

        if args.rank == 0:
            # FIXME: not exact global accuracy
            pos_acc = (TP.sum()/TP.size(0)).item()
            noise_acc = (FP.sum()/FP.size(0)).item()
            manager.monitor.update('Pos Acc', pos_acc)
            manager.monitor.update('Noise Acc', noise_acc)
            manager.writer.add_scalar(
                'Acc/positive', pos_acc, manager.step + (i+1) % fold)
            manager.writer.add_scalar(
                'Acc/noise', noise_acc, manager.step + (i+1) % fold)

        loss = loss / fold
        if not isinstance(norm_size, torch.Tensor):
            norm_size = input_lengths.new_tensor(
                int(norm_size), device=args.gpu)
        else:
            norm_size = norm_size.to(device=args.gpu, dtype=torch.long)

        normalized_loss = loss.detach() * norm_size
        if 'databalance' in args and args.databalance:
            '''
            get current global size
            efficiently, we can set t_batch_size=args.batch_size, 
            but current impl is more robust
            '''
            dist.all_reduce(norm_size)
            loss.data = normalized_loss * (world_size / norm_size)
        else:
            norm_size *= world_size

        scaler.scale(loss).backward()

        dist.all_reduce(normalized_loss)
        detach_loss += normalized_loss.float()
        n_batch += norm_size

        return detach_loss, n_batch

    utils.check_parser(args, ['grad_accum_fold', 'n_steps',
                              'print_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm'])

    model = manager.model
    scheduler = manager.scheduler
    optimizer = scheduler.optimizer
    optimizer.zero_grad()
    enableAMP = args.amp
    scaler = GradScaler(enabled=enableAMP)
    grad_norm = args.grad_norm

    world_size = dist.get_world_size()
    fold = args.grad_accum_fold
    assert fold >= 1
    detach_loss = 0.0
    n_batch = 0
    for i, minibatch in tqdm(enumerate(trainloader), desc=f'Epoch {manager.epoch} | train',
                             unit='batch', total=fold*args.n_steps, disable=(args.gpu != 0), leave=False):

        features, input_lengths, labels, label_lengths = minibatch
        features, labels, input_lengths, label_lengths = features.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        if manager.specaug is not None:
            features, input_lengths = manager.specaug(features, input_lengths)

        # update every fold times and drop the last few batches (number of which <= fold)
        if fold == 1 or (i+1) % fold == 0:
            detach_loss, n_batch = _go_step(detach_loss, n_batch)

            if grad_norm > 0.0:
                if enableAMP:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm, error_if_nonfinite=False)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            manager.step += 1
            global_step = manager.step
            scheduler.update_lr(global_step)

            # average for logging
            dist.all_reduce(detach_loss)
            detach_loss /= n_batch
            # measure accuracy and record loss; item() can sync all processes.
            tolog = {
                'loss': detach_loss.item(),
                'lr': scheduler.lr_cur
            }

            # update tensorboard
            if args.rank == 0:
                manager.writer.add_scalar(
                    'loss/train_loss', tolog['loss'], global_step)
                manager.writer.add_scalar(
                    'lr', tolog['lr'], global_step)

            # update monitor
            manager.monitor.update({
                'train:loss': tolog['loss'],
                'train:lr': tolog['lr']
            })

            n_time = (i+1)//fold

            if n_time == args.n_steps or (args.debug and n_time >= 20):
                dist.barrier()
                break

            # reset accumulated loss
            detach_loss -= detach_loss
            n_batch -= n_batch
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                detach_loss, n_batch = _go_step(detach_loss, n_batch)


def build_model(args, configuration: dict, dist: bool = True) -> DiscTransducerTrainer:
    def _load_and_immigrate(orin_dict_path: str, str_src: str, str_dst: str) -> OrderedDict:
        if not os.path.isfile(orin_dict_path):
            raise FileNotFoundError(f"{orin_dict_path} is not a valid file.")

        checkpoint = torch.load(orin_dict_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_state_dict[k.replace(str_src, str_dst)] = v
        del checkpoint
        return new_state_dict

    if 'DiscTransducerTrainer' not in configuration:
        configuration['DiscTransducerTrainer'] = {}

    assert 'ctc-sampler' in configuration
    ctc_config = configuration['ctc-sampler']
    assert 'pretrain-config' in ctc_config
    assert 'pretrain-check' in ctc_config
    with open(ctc_config['pretrain-config'], 'r') as fi:
        ctc_setting = json.load(fi)
    ctc_model = ctc_builder(None, ctc_setting, dist=False, wrapper=False)
    ctc_model.load_state_dict(_load_and_immigrate(
        ctc_config['pretrain-check'], 'module.am.', ''))
    ctc_model.eval()
    for param in ctc_model.parameters():
        param.requires_grad = False

    assert 'searcher' in configuration
    encoder, decoder, joint = rnnt_builder(
        args, configuration, dist=False, verbose=True, wrapped=False)

    labels = [str(i) for i in range(ctc_model.classifier.out_features)]
    searcher = CTCBeamDecoder(
        labels, log_probs_input=True, **configuration['searcher'])

    model = DiscTransducerTrainer(
        encoder, decoder, joint, ctc_model, searcher, **configuration['DiscTransducerTrainer'])

    if not dist:
        setattr(model, 'requires_slice', True)
        return model

    # make batchnorm synced across all processes
    model = utils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    setattr(model, 'requires_slice', True)

    return model


if __name__ == "__main__":
    parser = utils.BasicDDPParser()
    parser.add_argument("--gen", action="store_true",
                        help="Generate noise samples, used with --sample_path")
    parser.add_argument("--sample_path", type=str,
                        help="Path to generated samples.")
    args = parser.parse_args()

    utils.setPath(args)
    utils.main_spawner(args, main_worker)
