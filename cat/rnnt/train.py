# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""
Transducer trainer.
"""

from ..shared import Manager
from ..shared import coreutils as utils
from ..shared import encoder as tn_zoo
from ..shared import decoder as pn_zoo
from ..shared.layer import TimeReduction
from ..shared import SpecAug
from ..shared.decoder import AbsDecoder
from ..shared.data import (
    SpeechDatasetPickle,
    sortedPadCollateTransducer
)
from . import joint as joint_zoo
from .joint import (
    PackedSequence,
    AbsJointNet,
    DenormalJointNet
)

import os
import argparse
from collections import OrderedDict
from warp_rnnt import rnnt_loss as RNNTLoss
import warp_rnnt
if warp_rnnt.__version__ >= '0.7.1':
    from warp_rnnt import fused_rnnt_loss_ as RNNTFusedLoss
from warp_rna import rna_loss as RNALoss
from typing import Union

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


class TransducerTrainer(nn.Module):
    def __init__(self,
                 encoder: nn.Module = None,
                 decoder: AbsDecoder = None,
                 jointnet: AbsJointNet = None,
                 compact: bool = False,
                 fused: bool = False,
                 isrna: bool = False,
                 time_reduction: int = 1,
                 decoder_mask_range: float = 0.1,
                 num_decoder_mask: int = -1,
                 bos_id: int = 0):
        super().__init__()
        if isrna:
            assert not compact and not fused, f"RNA Loss currently doesn't support compact and fused mode yet."

        if fused and not jointnet.is_normalize_separated:
            raise RuntimeError(
                f"TransducerTrainer: {jointnet.__class__.__name__} is conflict with fused=True")
        if fused and not compact:
            print(
                "TransducerTrainer: setting fused=True and compact=False is conflict. Force compact=True")
            compact = True

        self.isfused = fused
        self._compact = compact
        self.isrna = isrna

        self.encoder = encoder
        self.decoder = decoder
        self.joint = jointnet

        assert isinstance(time_reduction, int) and time_reduction >= 1
        if time_reduction == 1:
            self._t_reduction = None
        else:
            self._t_reduction = TimeReduction(time_reduction)

        if num_decoder_mask != -1:
            self._pn_mask = SpecAug(time_mask_width_range=decoder_mask_range,
                                    num_time_mask=num_decoder_mask, apply_freq_mask=False, apply_time_warp=False)
        else:
            self._pn_mask = None

        self.bos_id = bos_id

    def train(self: 'TransducerTrainer', mode: bool = True) -> 'TransducerTrainer':
        super().train(mode=mode)
        if self.encoder.freeze:
            self.encoder.eval()
        if self.decoder.freeze:
            self.decoder.eval()
        return self

    def impl_forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        targets = targets.to(inputs.device, non_blocking=True)

        output_encoder, o_lens = self.encoder(inputs, input_lengths)
        o_lens = o_lens.to(torch.long)
        # introduce time reduction layer
        if self._t_reduction is not None:
            output_encoder, o_lens = self._t_reduction(output_encoder, o_lens)

        padded_targets = torch.cat(
            [targets.new_full((targets.size(0), 1), self.bos_id), targets], dim=-1)
        output_decoder, _ = self.decoder(padded_targets)

        if self._pn_mask is not None:
            output_decoder, _ = self._pn_mask(output_decoder, target_lengths+1)

        if self._compact:
            packed_enc = PackedSequence(output_encoder, o_lens)
            packed_dec = PackedSequence(output_decoder, target_lengths+1)
            if self.isfused:
                joint_out = self.joint.impl_forward(
                    packed_enc, packed_dec)
            else:
                joint_out = self.joint(packed_enc, packed_dec)
            targets = PackedSequence(targets, target_lengths).data
        else:
            joint_out = self.joint(output_encoder, output_decoder)

        return joint_out, targets, o_lens, target_lengths

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        joint_out, targets, o_lens, target_lengths = self.impl_forward(
            inputs, targets, input_lengths, target_lengths)

        if isinstance(joint_out, tuple):
            joint_out = joint_out[0]

        with autocast(enabled=False):
            if self.isrna:
                loss = RNALoss(joint_out.float(), targets.to(dtype=torch.int),
                               o_lens.to(device=joint_out.device,
                                         dtype=torch.int),
                               target_lengths.to(device=joint_out.device, dtype=torch.int), reduction='mean')
            elif self.isfused:
                loss = RNNTFusedLoss(joint_out.float(), targets.to(dtype=torch.int32),
                                     o_lens.to(device=joint_out.device,
                                               dtype=torch.int32),
                                     target_lengths.to(device=joint_out.device, dtype=torch.int32), reduction='mean')
            else:
                loss = RNNTLoss(joint_out.float(), targets.to(dtype=torch.int32),
                                o_lens.to(device=joint_out.device, dtype=torch.int32), target_lengths.to(
                                    device=joint_out.device, dtype=torch.int32),
                                reduction='mean', gather=True, compact=self._compact)

        return loss


@torch.no_grad()
def build_model(args, configuration: dict, dist: bool = True, verbose: bool = True, wrapped: bool = True) -> Union[nn.Module, nn.parallel.DistributedDataParallel]:
    def _load_and_immigrate(orin_dict_path: str, str_src: str, str_dst: str) -> OrderedDict:
        if not os.path.isfile(orin_dict_path):
            raise FileNotFoundError(f"{orin_dict_path} is not a valid file.")

        checkpoint = torch.load(orin_dict_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_state_dict[k.replace(str_src, str_dst)] = v
        del checkpoint
        return new_state_dict

    def _build(config: dict, module: str) -> nn.Module:
        assert 'kwargs' in config

        if module == 'encoder':
            zoo = tn_zoo
        elif module == 'decoder':
            zoo = pn_zoo
            if "type" not in config:
                # compatibility to older config
                config["type"] = "LSTMPredictNet"
        elif module == 'joint':
            """
                The joint network accept the concatence of outputs of the 
                encoder and decoder. So the input dimensions MUST match that.
            """
            zoo = joint_zoo
            if 'type' not in config:
                config['type'] = 'JointNet'

        else:
            raise ValueError(f"Unknow module: {module}")

        _model = getattr(zoo, config['type'])(
            **config['kwargs'])  # type: AbsJointNet

        if "pretrained" in config:
            if module == "encoder":
                prefix = 'module.am.'
            elif module == "decoder":
                prefix = 'module.lm.'
            else:
                raise RuntimeError(
                    "Unknown module with 'pretrained' option: {}".format(module))

            init_sum = sum(param.data.sum()
                           for param in _model.parameters())
            state_dict = _load_and_immigrate(
                config['pretrained'], prefix, '')
            _model.load_state_dict(state_dict, strict=False)
            if sum(param.data.sum()for param in _model.parameters()) == init_sum:
                utils.highlight_msg(
                    f"WARNING: It seems {module} pretrained model is not properly loaded.")

        if 'freeze' in config and config['freeze']:
            if 'pretrained' not in config:
                raise RuntimeError(
                    "freeze=True while 'pretrained' is empty is not allowed. In {} init".format(module))

            for param in _model.parameters():
                param.requires_grad = False

            setattr(_model, 'freeze', True)
        else:
            setattr(_model, 'freeze', False)

        if verbose and args.rank == 0:
            if 'pretrained' not in config:
                _path = ''
            else:
                _path = config['pretrained']
            print("{:<8}: freeze={:<5} | loaded from {}".format(
                module.upper(), str(_model.freeze), _path))
            del _path
        return _model

    assert 'encoder' in configuration
    assert 'decoder' in configuration
    assert 'joint' in configuration

    encoder = _build(configuration['encoder'], 'encoder')
    decoder = _build(configuration['decoder'], 'decoder')
    jointnet = _build(configuration['joint'], 'joint')
    if all(_model.freeze for _model in [encoder, decoder, jointnet]):
        raise RuntimeError("It's illegal to freeze all parts of Transducer.")

    is_part_freeze = not all(not _model.freeze for _model in [
                             encoder, decoder, jointnet])

    if not wrapped:
        return encoder, decoder, jointnet

    # for compatible of old settings
    if 'transducer' in configuration:
        transducer_kwargs = configuration["transducer"]     # type: dict
    else:
        transducer_kwargs = {}

    model = TransducerTrainer(encoder=encoder, decoder=decoder,
                              jointnet=jointnet, **transducer_kwargs)

    if not dist:
        if is_part_freeze:
            setattr(model, 'requires_slice', True)
        return model

    # make batchnorm synced across all processes
    model = utils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    if is_part_freeze:
        setattr(model, 'requires_slice', True)
    return model


def RNNTParser():
    parser = utils.BasicDDPParser("RNN-Transducer training")
    return parser


def main(args: argparse.Namespace = None):
    if args is None:
        parser = RNNTParser()
        args = parser.parse_args()

    utils.setPath(args)
    utils.main_spawner(args, main_worker)
