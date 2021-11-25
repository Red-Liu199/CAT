# copyright Tsinghua University
# Author: Huahuan Zheng (maxwellzh@outlook.com)

from ..shared import Manager
from ..shared import coreutils as utils
from ..shared import encoder as tn_zoo
from ..shared import decoder as pn_zoo
from .beam_search_transducer import TransducerBeamSearcher
from ..shared.decoder import AbsDecoder
from .train import build_model as rnnt_builder
from ..lm import lm_builder
from ..shared.decoder import NGram
from . import JointNet, SimJointNet
from ..shared.data import (
    SpeechDatasetPickle,
    sortedPadCollateTransducer
)
from . import PackedSequence
import warp_rnnt

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
                 encoder: nn.Module,
                 decoder: AbsDecoder,
                 joint: JointNet,
                 den_lm: AbsDecoder,
                 searcher: TransducerBeamSearcher) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint

        self.den_lm = den_lm
        self.searcher = searcher
        self._pad = nn.ConstantPad1d((1, 0), 0)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor):

        targets = targets.to(inputs.device, non_blocking=True)
        padded_targets = self._pad(targets)

        output_encoder, encoder_lens = self.encoder(inputs, input_lengths)

        with torch.no_grad():
            # FIXME: magic codes for fixing the bug:
            # "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
            # ...This error indicates that your module has parameters that were not used in producing loss."
            for param in self.joint.parameters():
                param.requires_grad = False

            encoder_lens = encoder_lens.to(torch.long)
            N = output_encoder.size(0)
            difflist = []
            negsamples = []
            for n in range(N):
                hyps = self.searcher.raw_decode(
                    output_encoder[n:n+1, :encoder_lens[n]])
                if hyps[0].pred == padded_targets[n][:target_lengths[n]+1]:
                    continue
                difflist.append(n)
                negsamples.append(hyps[0].pred)

            M = len(difflist)
            if M == 0:
                with torch.enable_grad():
                    loss = 0.0
                    # hack way to skip parameter updating
                    for param in self.parameters():
                        if param.requires_grad:
                            loss += 0.0 * param.sum()
                    return loss

            possamples = [padded_targets[d, :1+target_lengths[d]]
                          for d in difflist]
            negsamples = [inputs.new_tensor(_neg_sample)
                          for _neg_sample in negsamples]

            merge_samples = possamples + negsamples
            # length include <s>
            merge_lengths = input_lengths.new_tensor(
                [sample.size(0) for sample in merge_samples], dtype=torch.long)
            # [2M, U]
            merge_samples = utils.pad_list(merge_samples)
            # [2M,]
            den_score = self.den_lm.score(merge_samples, merge_lengths)

            for param in self.joint.parameters():
                param.requires_grad = True

        # [2M, T, H]
        sampled_encoder_out = output_encoder[difflist*2, :, :]
        # [2M, ]
        sampled_encoder_lens = encoder_lens[difflist*2]
        # [2M, U, H]
        sampled_decoder_out, _ = self.decoder(
            merge_samples, input_lengths=merge_lengths)

        packed_decoder_out = PackedSequence(
            sampled_decoder_out, merge_lengths)
        packed_encoder_out = PackedSequence(
            sampled_encoder_out, sampled_encoder_lens)

        joint_out = self.joint.skip_softmax_forward(
            packed_encoder_out, packed_decoder_out)
        target_lengths = (
            merge_lengths - 1).to(device=joint_out.device, dtype=torch.int32)
        targets = gather.cat(merge_samples[:, 1:].unsqueeze(
            2).to(torch.float), target_lengths).to(dtype=torch.int32)

        with autocast(enabled=False):
            f_joint = joint_out.float()
            frame_lengths = sampled_encoder_lens.to(
                device=joint_out.device, dtype=torch.int32)
            # use logit instead of log probs to cal un-normalized "cost"
            # [2M, ]
            p_rnnt_costs = warp_rnnt.rnnt_loss(
                f_joint, targets, frame_lengths, target_lengths,
                reduction='none', gather=True, compact=True)
            with torch.no_grad():
                # [2M, ]
                q_rnnt_costs = warp_rnnt.fused_rnnt_loss_(
                    f_joint, targets, frame_lengths,
                    target_lengths, reduction=None)
        # assume lm weight \lambda=1.0
        # log p(Y) = logP_rnnt(Y|X)+logP_lm(Y), [2M, ]
        # type: torch.FloatTensor
        p_y_all = -p_rnnt_costs+den_score
        q_y_all = -q_rnnt_costs
        # [2M, ]   p(x) + q(x)
        nce_denominator = torch.logaddexp(p_y_all, q_y_all)
        # p(x)/(p(x)+q(x)) & q(x)/(p(x)+q(x))
        concat_logits = torch.cat([
            p_y_all[:M], q_y_all[M:]
        ], dim=0) - nce_denominator

        nce_loss = -concat_logits.sum(dim=0) / inputs.size(0)
        # re-normalize with batch size
        return nce_loss

    def train(self, mode: bool = True):
        super().train(mode)
        self.den_lm.eval()
        return


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
    assert 'pretrain-check' in disc_settings

    assert 'den-lm' in configuration
    den_lm_settins = configuration['den-lm']
    assert 'pretrain-config' in den_lm_settins

    assert 'searcher' in configuration

    with open(disc_settings['pretrain-config'], 'r') as fi:
        rnnt_setting = json.load(fi)
    encoder, decoder, joint = rnnt_builder(
        None, rnnt_setting, dist=False, verbose=False, wrapped=False)

    with open(den_lm_settins['pretrain-config'], 'r') as fi:
        dlm_config = json.load(fi)
    den_lm = lm_builder(None, dlm_config, dist=False, wrapper=False)
    for param in den_lm.parameters():
        param.requires_grad = False

    searcher = TransducerBeamSearcher(
        decoder, joint, **configuration['searcher'])

    model = DiscTransducerTrainer(encoder, decoder, joint, den_lm, searcher)
    transducer = _load_and_immigrate(
        configuration['DiscTransducerTrainer']['pretrain-check'], 'module.', '')

    if not isinstance(den_lm, NGram):
        transducer.update(_load_and_immigrate(
            den_lm_settins['pretrain-check'], 'module.lm.', 'den_lm.'))
    model.load_state_dict(transducer, strict=True)

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
