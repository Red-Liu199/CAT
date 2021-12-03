# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)

from ..shared import Manager
from ..shared import coreutils as utils
from .beam_search_transducer import TransducerBeamSearcher
from ..shared.decoder import AbsDecoder
from ..shared.encoder import AbsEncoder
from .train import build_model as rnnt_builder
from ..ctc.train import build_model as ctc_builder
from ..lm import lm_builder
from ..shared.decoder import NGram
from .joint import (
    PackedSequence,
    AbsJointNet,
    DenormalJointNet
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
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    utils.SetRandomSeed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    manager = Manager(SpeechDatasetPickle,
                      sortedPadCollateTransducer(), args, build_model)

    # training
    manager.run(args)


class DiscTransducerTrainer(nn.Module):
    def __init__(self,
                 encoder: AbsEncoder,
                 decoder: AbsDecoder,
                 joint: AbsJointNet,
                 ctc_sampler: AbsEncoder,
                 searcher: CTCBeamDecoder,
                 beta: float = 0.6) -> None:
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

        # calculate the noise q(Y|X)
        with torch.no_grad(), autocast(enabled=False):
            # [N, T, V] -> [T, N, V]
            noise_xs = noise_xs.transpose(0, 1)
            noise_log_probs = - \
                self.ctc_loss(noise_xs.float().log_softmax(
                    dim=-1), ys, lnx, ly)

        with autocast(enabled=False):
            ys = gather.cat(ys.unsqueeze(2).float(), ly).squeeze(0)
            dist_log_probs = -warp_rnnt.rnnt_loss(
                model_xs.float(), ys.to(torch.int), lx, ly,
                reduction='none', gather=True, compact=True) + self._beta * ly

        q_nu = math.log(nu) + noise_log_probs
        if is_noise:
            return q_nu - torch.logaddexp(dist_log_probs, q_nu)
        else:
            return dist_log_probs - torch.logaddexp(dist_log_probs, q_nu)

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

        return -(pos_logp.mean() + noise_logp.mean())

    def train(self, mode: bool = True):
        super().train(mode)
        self.sampler.eval()
        return


class MMITransducerTrainer(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: AbsDecoder,
                 joint: DenormalJointNet,
                 searcher: TransducerBeamSearcher) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint

        self.searcher = searcher
        self._pad = nn.ConstantPad1d((1, 0), 0)
        self._random_mask = True

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor):

        targets = targets.to(inputs.device, non_blocking=True)
        output_encoder, encoder_lens = self.encoder(inputs, input_lengths)
        encoder_lens = encoder_lens.to(torch.long, non_blocking=True)
        # numerator
        # [N, ]
        numerator_cost = self.cal_p(
            output_encoder, targets, encoder_lens, target_lengths)

        if self._random_mask:
            K = self.searcher.beam_size
            # [N, U] -> [N, 1, U] -> [N, K, U]
            unsqz_targets = targets.unsqueeze(1).expand(-1, K, -1)
            # mask 20% tokens
            mask = torch.rand(unsqz_targets.size(),
                              device=targets.device) < 0.2
            # random token, <eos>=0
            values = torch.randint_like(
                unsqz_targets, 1, output_encoder.size(-1))
            # [N, K, U]
            den_samples = unsqz_targets.masked_scatter(mask, values)
            # [N, K, U] -> [NK, U]
            den_samples = den_samples.view(-1, den_samples.size(-1))
            # [N, K] -> [NK, ]
            den_samples_lengths = target_lengths.unsqueeze(
                1).expand(-1, K).contiguous().view(-1)

            # [N, K, T, V]
            sampled_encoder_out = output_encoder.unsqueeze(
                1).expand(-1, K, -1, -1)
            # [N, K, T, V] -> [NK, T, V]
            sampled_encoder_out = sampled_encoder_out.contiguous().view(
                -1, *(sampled_encoder_out.size()[-2:]))
            # [N, K] -> [NK, ]
            sampled_encoder_lens = encoder_lens.unsqueeze(
                1).expand(-1, K).contiguous().view(-1)
            # [NK, ]
            denominator_cost = self.cal_p(
                sampled_encoder_out, den_samples, sampled_encoder_lens, den_samples_lengths)

            # [NK, ] -> [N, K] -> [N, 1+K]
            denominator_cost = torch.cat(
                [numerator_cost.unsqueeze(1), denominator_cost.view(-1, K)], dim=1)

            mmi_loss = -(numerator_cost -
                         torch.logsumexp(denominator_cost, dim=1))

            return mmi_loss.mean()

        else:
            with torch.no_grad():
                N = output_encoder.size(0)
                difflist = []
                den_samples = []
                cumsum = [0]
                batch_samples = self.searcher.batch_alsd(
                    output_encoder, encoder_lens)
                for n in range(N):
                    hyps = batch_samples[n]
                    difflist += [n] * len(hyps)
                    cumsum.append(len(difflist))
                    for hypo in hyps:
                        if torch.equal(targets[n, :target_lengths[n]], hypo.pred[1:]):
                            difflist[-1].pop()
                            cumsum[-1] -= 1
                            continue
                        den_samples.append(hypo.pred[1:])
                # for n in range(N):
                #     hyps = self.searcher.raw_decode(
                #         host_encoder_out[n:n+1, :encoder_lens[n]])
                #     difflist += [n] * len(hyps)
                #     cumsum.append(len(difflist))
                #     for hypo in hyps:
                #         den_samples.append(hypo.pred[1:])

                M = len(difflist)   # number of denominator of whole batch
                # length exnclude <s>
                den_samples_lengths = input_lengths.new_tensor(
                    [sample.size(0) for sample in den_samples], dtype=torch.long)
                # [M, U]
                den_samples = utils.pad_list(den_samples).to(inputs.device)

            # denominator
            # [M, T, V]
            sampled_encoder_out = output_encoder[difflist, :, :]
            # [M, ]
            sampled_encoder_lens = encoder_lens[difflist]
            # [M, ]
            denominator_cost = self.cal_p(
                sampled_encoder_out, den_samples, sampled_encoder_lens, den_samples_lengths)

            mmi_loss = []
            for n in range(N):
                mmi_loss.append(
                    -(numerator_cost[n]-torch.logsumexp(torch.cat([
                        denominator_cost[cumsum[n]:cumsum[n+1]
                                         ], numerator_cost[n].view(1)
                    ]), dim=0)))

            return sum(mmi_loss)/N

    def cal_p(self, xs: torch.Tensor, labels: torch.tensor, lx: torch.LongTensor, ly: torch.LongTensor):
        padded_y = self._pad(labels)
        pn_out, _ = self.decoder(padded_y, input_lengths=ly+1)

        lx = lx.to(device=xs.device, dtype=torch.int)
        ly = ly.to(device=xs.device, dtype=torch.int)
        packed_x = PackedSequence(xs, lx)
        packed_y = PackedSequence(pn_out, ly+1)
        try:
            joint_out = self.joint(packed_x, packed_y)
        except RuntimeError as re:
            print(re)
            print(packed_x.data.size())
            print(packed_x.batch_sizes)
            print(packed_y.data.size())
            print(packed_y.batch_sizes)
            exit(1)
        # [N, U] -> [SN, ]
        labels = gather.cat(labels.unsqueeze(
            2).to(torch.float), ly).to(dtype=torch.int32).squeeze(1)
        with autocast(enabled=False):
            rnnt_cost = warp_rnnt.rnnt_loss(
                joint_out, labels, lx, ly, gather=True, compact=True)
        return -rnnt_cost

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return


def build_mmi_model(args, configuration: dict, dist: bool = True) -> DiscTransducerTrainer:
    def _load_and_immigrate(orin_dict_path: str, str_src: str, str_dst: str) -> OrderedDict:
        if not os.path.isfile(orin_dict_path):
            raise FileNotFoundError(f"{orin_dict_path} is not a valid file.")

        checkpoint = torch.load(orin_dict_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_state_dict[k.replace(str_src, str_dst)] = v
        del checkpoint
        return new_state_dict

    assert 'TransducerTrainer' in configuration
    disc_settings = configuration['TransducerTrainer']
    assert 'pretrain-config' in disc_settings
    assert 'pretrain-lm-check' in disc_settings

    assert 'searcher' in configuration

    with open(disc_settings['pretrain-config'], 'r') as fi:
        rnnt_setting = json.load(fi)
    encoder, decoder, joint = rnnt_builder(
        None, rnnt_setting, dist=False, verbose=False, wrapped=False)

    utils.distprint(
        f"> MMI Transducer: PN initializes from {disc_settings['pretrain-lm-check']}", args.gpu)
    decoder.load_state_dict(_load_and_immigrate(
        disc_settings['pretrain-lm-check'], 'module.lm.', ''))
    for param in decoder.parameters():
        param.requires_grad = False
    decoder.eval()
    # cpu_decoder = copy.deepcopy(decoder.to('cpu'))
    # cpu_decoder.eval()
    searcher = TransducerBeamSearcher(
        decoder, joint, **configuration['searcher'])
    assert isinstance(joint, DenormalJointNet)
    model = MMITransducerTrainer(encoder, decoder, joint, searcher)

    if not dist:
        return model

    # make batchnorm synced across all processes
    model = utils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


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

    assert 'DiscTransducerTrainer' in configuration
    disc_settings = configuration['DiscTransducerTrainer']
    assert 'pretrain-config' in disc_settings
    assert 'pretrain-lm-check' in disc_settings

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
    with open(disc_settings['pretrain-config'], 'r') as fi:
        rnnt_setting = json.load(fi)
    encoder, decoder, joint = rnnt_builder(
        None, rnnt_setting, dist=False, verbose=False, wrapped=False)

    utils.distprint("> Initialize prediction network from {}".format(
        disc_settings['pretrain-lm-check']), args.gpu)
    decoder.load_state_dict(_load_and_immigrate(
        disc_settings['pretrain-lm-check'], 'module.lm.', ''))
    for param in decoder.parameters():
        param.requires_grad = False

    labels = [str(i) for i in range(ctc_model.classifier.out_features)]
    searcher = CTCBeamDecoder(
        labels, log_probs_input=True, **configuration['searcher'])

    model = DiscTransducerTrainer(encoder, decoder, joint, ctc_model, searcher)

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
