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
from ..shared.data import (
    SpeechDataset,
    SpeechDatasetPickle,
    sortedPadCollateTransducer
)
from . import (
    PackedSequence,
    JointNet,
    SimJointNet
)

import os
import argparse
from collections import OrderedDict
from warp_rnnt import rnnt_loss as RNNTLoss
import warp_rnnt
if warp_rnnt.__version__ >= '0.7.0':
    from warp_rnnt import fused_rnnt_loss as RNNTFusedLoss
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

    if args.h5py:
        Dataset = SpeechDataset
    else:
        Dataset = SpeechDatasetPickle

    manager = Manager(Dataset, sortedPadCollateTransducer(), args, build_model)

    if args.update_bn:
        assert args.resume is not None, "invalid behavior"

        utils.update_bn(manager.trainloader, args, manager)
        updated_check = args.resume.replace('.pt', '_bn.pt')
        manager.save(updated_check)
        utils.distprint(
            f"> Save updated model at {updated_check}", args.rank)
        return

    # training
    manager.run(args)


class TransducerTrainer(nn.Module):
    def __init__(self,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 jointnet: nn.Module = None,
                 compact: bool = False,
                 fused: bool = False,
                 time_reduction: int = 1,
                 decoder_mask_range: float = 0.1,
                 num_decoder_mask: int = -1,
                 bos_id: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = jointnet
        self._compact = compact

        if fused and not hasattr(self.joint, 'skip_softmax_forward'):
            raise RuntimeError(
                f"class {self.joint.__name__} doesn't have method \'skip_softmax_forward\' implemented.")

        self.isfused = fused
        if self.isfused and not self._compact:
            print(
                "TransducerTrainer: setting isfused=True and compact=False is conflict. Force compact=True")
            self._compact = True
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

    def impl_forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        targets = targets.to(inputs.device, non_blocking=True)

        output_encoder, o_lens = self.encoder(inputs, input_lengths)
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
                joint_out = self.joint.skip_softmax_forward(
                    packed_enc, packed_dec)
            else:
                joint_out = self.joint(packed_enc, packed_dec)
            targets = PackedSequence(targets, target_lengths).data
        else:
            if self.isfused:
                joint_out = self.joint.skip_softmax_forward(
                    output_encoder, output_decoder)
            else:
                joint_out = self.joint(output_encoder, output_decoder)

        return joint_out, targets, o_lens, target_lengths

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        joint_out, targets, o_lens, target_lengths = self.impl_forward(
            inputs, targets, input_lengths, target_lengths)

        if isinstance(joint_out, tuple):
            joint_out = joint_out[0]

        with autocast(enabled=False):
            if self.isfused:
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
def build_model(args, configuration: dict, dist: bool = True, verbose: bool = True) -> Union[nn.Module, nn.parallel.DistributedDataParallel]:
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

        settings = config['kwargs']     # type: dict
        if module == 'encoder':
            _model = getattr(tn_zoo, config['type'])(
                **settings)  # type: nn.Module

        elif module == 'decoder':
            if "type" not in config:
                # compatibility to older config
                config["type"] = "LSTMPredictNet"
            _model = getattr(pn_zoo, config['type'])(
                **settings)  # type: nn.Module

        elif module == 'joint':
            """
                The joint network accept the concatence of outputs of the 
                encoder and decoder. So the input dimensions MUST match that.
            """
            if 'type' not in config:
                AbsNet = JointNet
            else:
                AbsNet = eval(config['type'])  # type: JointNet
            _model = AbsNet(**settings)
        else:
            raise ValueError(f"Unknow module: {module}")

        if "pretrained" in config:
            if module == "encoder":
                # 'module.infer.' will be deprecated soon, -> 'module.am.'
                prefix = 'module.infer.'
            elif module == "decoder":
                prefix = 'module.lm.'
            else:
                raise RuntimeError(
                    "Unknown module with 'pretrained' option: {}".format(module))

            del _model.classifier
            init_sum = sum(param.data.sum()
                           for param in _model.parameters())
            state_dict = _load_and_immigrate(
                config['pretrained'], prefix, '')
            _model.load_state_dict(state_dict, strict=False)
            if sum(param.data.sum()for param in _model.parameters()) == init_sum:
                utils.highlight_msg(
                    "WARNING: It seems decoder pretrained model is not properly loaded.")

        if module in ['encoder', 'decoder']:
            # FIXME: this is a hack, since we just feed the hidden output into joint network
            _model.classifier = nn.Identity()

        # NOTE (Huahuan): In a strict sense, we should avoid invoke model.train() if we want to freeze the model
        #                 ...for which would enable the operations like dropout during training.
        if 'freeze' in config and config['freeze']:
            if 'pretrained' not in config:
                raise RuntimeError(
                    "freeze=True while 'pretrained' is empty is not allowed. In {} init".format(module))

            for name, param in _model.named_parameters():
                # NOTE: we only freeze those loaded parameters
                if name in state_dict and param.requires_grad:
                    param.requires_grad = False

            setattr(_model, 'freeze', True)
        else:
            setattr(_model, 'freeze', False)

        if verbose and args.rank == 0:
            if 'pretrained' not in config:
                _path = ''
            else:
                _path = config['pretrained']
            print("{:<8}: freeze={} | loaded from {}".format(
                module.upper(), _model.freeze, _path))
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

    # for compatible of old settings
    if 'transducer' in configuration:
        transducer_kwargs = configuration["transducer"]     # type: dict
    else:
        transducer_kwargs = {}

    model = TransducerTrainer(encoder=encoder, decoder=decoder,
                              jointnet=jointnet, **transducer_kwargs)

    if not all(not _model.freeze for _model in [encoder, decoder, jointnet]):
        setattr(model, 'requires_slice', True)

    if not dist:
        return model

    # make batchnorm synced across all processes
    model = utils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


def main():
    parser = utils.BasicDDPParser()
    parser.add_argument("--h5py", action="store_true",
                        help="Load data with H5py, defaultly use pickle (recommended).")

    args = parser.parse_args()

    utils.setPath(args)

    utils.main_spawner(args, main_worker)
