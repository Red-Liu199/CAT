# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""
Transducer trainer.
"""

from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as tn_zoo
from ..shared import decoder as pn_zoo
from ..shared.layer import TimeReduction
from ..shared import SpecAug
from ..shared.data import (
    KaldiSpeechDataset,
    sortedPadCollateASR
)
from . import joint as joint_zoo
from .joint import (
    PackedSequence,
    JointNet,
    AbsJointNet
)

import os
import gather
import argparse
from collections import OrderedDict
from warp_rnnt import rnnt_loss as RNNTLoss
from warp_rnnt import fused_rnnt_loss_ as RNNTFusedLoss
from typing import Union, Optional, Tuple

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

    manager = Manager(KaldiSpeechDataset,
                      sortedPadCollateASR(), args, build_model)

    # training
    manager.run(args)


class TransducerTrainer(nn.Module):
    def __init__(
        self,
        encoder: tn_zoo.AbsEncoder = None,
        decoder: pn_zoo.AbsDecoder = None,
        jointnet: AbsJointNet = None,
        # enable compact layout, would consume less memory and fasten the computation of RNN-T loss.
        compact: bool = False,
        # enable fused mode for RNN-T loss computation, which would consume less memory but take a little more time
        fused: bool = False,
        # weight of ILME loss to conduct joint-training
        ilme_loss: Optional[float] = None,
        # insert sub-sampling layer between encoder and joint net
        time_reduction: int = 1,
        # add mask to decoder output, this specifies the range of mask
        decoder_mask_range: float = 0.1,
        # add mask to decoder output, this specifies the # mask
        num_decoder_mask: int = -1,
            bos_id: int = 0):
        super().__init__()

        if fused and not jointnet.is_normalize_separated:
            raise RuntimeError(
                f"TransducerTrainer: {jointnet.__class__.__name__} is conflict with fused=True")
        if fused and not compact:
            print(
                "TransducerTrainer: setting fused=True and compact=False is conflict. Force compact=True")
            compact = True

        self.ilme_weight = 0.0
        if ilme_loss is not None and ilme_loss != 0.0:
            if not isinstance(jointnet, JointNet):
                raise NotImplementedError(
                    f"TransducerTrainer: \n"
                    f"ILME loss joint training only support joint network 'JointNet', instead of {jointnet.__class__.__name__}")
            assert jointnet.fc_dec is not None, "TransducerTrainer: are you using jointnet with \"joint_mode='cat'\"? This is not supported yet."
            self.register_buffer(
                "_dummy_h_enc_ilme_loss",
                torch.zeros(1, 1, jointnet.fc_enc.in_features),
                persistent=False)
            self._ilme_criterion = nn.CrossEntropyLoss()
            self.ilme_weight = ilme_loss

        self.isfused = fused
        self._compact = compact

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
            # squeeze targets to 1-dim
            targets = PackedSequence(targets, target_lengths).data.squeeze(1)
        else:
            joint_out = self.joint(output_encoder, output_decoder)

        return joint_out, targets, o_lens, target_lengths, output_encoder, output_decoder

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        tupled_jointout = self.impl_forward(
            inputs, targets, input_lengths, target_lengths)

        joint_out, targets, o_lens, target_lengths = tupled_jointout[:4]

        loss = 0.0
        if self.ilme_weight != 0.0:
            # calculate the ILME ILM loss
            # h_decoder_out: (N, U+1, H)
            h_decoder_out = tupled_jointout[5]
            # ilm_log_probs: (N, 1, U, V) -> (N, U, V)
            ilm_log_probs = self.joint(
                self._dummy_h_enc_ilme_loss.expand(
                    h_decoder_out.size(0), -1, -1),
                h_decoder_out[:, :-1, :]).squeeze(1)
            # ilm_log_probs: (N, U, V) -> (\sum(U_i), V)
            ilm_log_probs = gather.cat(ilm_log_probs, target_lengths)
            if targets.dim() == 2:
                # normal layout -> compact layout
                # ilm_targets: (\sum{U_i}, )
                ilm_targets = torch.cat(
                    [targets[n, :target_lengths[n]] for n in range(h_decoder_out.size(0))], dim=0)
            elif targets.dim() == 1:
                ilm_targets = targets
            else:
                raise ValueError(
                    f"{self.__class__.__name__}: invalid dimension of targets '{targets.dim()}', expected 1 or 2.")

            loss += self.ilme_weight * \
                self._ilme_criterion(ilm_log_probs, ilm_targets)

        with autocast(enabled=False):
            if self.isfused:
                loss += RNNTFusedLoss(joint_out.float(), targets.to(dtype=torch.int32),
                                      o_lens.to(device=joint_out.device,
                                                dtype=torch.int32),
                                      target_lengths.to(device=joint_out.device, dtype=torch.int32), reduction='mean')
            else:
                loss += RNNTLoss(joint_out.float(), targets.to(dtype=torch.int32),
                                 o_lens.to(device=joint_out.device, dtype=torch.int32), target_lengths.to(
                    device=joint_out.device, dtype=torch.int32),
                    reduction='mean', gather=True, compact=self._compact)

        return loss


@torch.no_grad()
def build_model(
        cfg: dict,
        args: Optional[Union[argparse.Namespace, dict]] = None,
        dist: bool = True,
        wrapped: bool = True,
        verbose: bool = True) -> Union[nn.parallel.DistributedDataParallel, TransducerTrainer, Tuple[tn_zoo.AbsEncoder, pn_zoo.AbsDecoder, AbsJointNet]]:

    if args is not None:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        elif not isinstance(args, dict):
            raise ValueError(f"unsupport type of args: {type(args)}")

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
        elif module == 'joint':
            zoo = joint_zoo
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

            _model.load_state_dict(
                _load_and_immigrate(
                    config['pretrained'],
                    prefix, ''), strict=False)

        if 'freeze' in config and config['freeze']:
            if 'pretrained' not in config:
                raise RuntimeError(
                    "freeze=True while 'pretrained' is empty is not allowed. In {} init".format(module))
            _model.requires_grad_(False)
            setattr(_model, 'freeze', True)
        else:
            setattr(_model, 'freeze', False)

        if verbose and args['rank'] == 0:
            if 'pretrained' not in config:
                _path = ''
            else:
                _path = config['pretrained']
            print("{:<8}: freeze={:<5} | loaded from {}".format(
                module.upper(), str(_model.freeze), _path))
            del _path
        return _model

    assert 'encoder' in cfg
    assert 'decoder' in cfg
    assert 'joint' in cfg
    verbose = False if args is None else verbose

    encoder = _build(cfg['encoder'], 'encoder')
    decoder = _build(cfg['decoder'], 'decoder')
    jointnet = _build(cfg['joint'], 'joint')
    if all(_model.freeze for _model in [encoder, decoder, jointnet]):
        raise RuntimeError("It's illegal to freeze all parts of Transducer.")

    is_part_freeze = not all(not _model.freeze for _model in [
                             encoder, decoder, jointnet])

    if not wrapped:
        return encoder, decoder, jointnet

    # for compatible of old settings
    if 'transducer' in cfg:
        transducer_kwargs = cfg["transducer"]     # type: dict
    else:
        transducer_kwargs = {}

    model = TransducerTrainer(encoder=encoder, decoder=decoder,
                              jointnet=jointnet, **transducer_kwargs)

    if not dist:
        if is_part_freeze:
            setattr(model, 'requires_slice', True)
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args['gpu'])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args['gpu']])
    if is_part_freeze:
        setattr(model, 'requires_slice', True)
    return model


def RNNTParser():
    parser = coreutils.basic_trainer_parser("RNN-Transducer training")
    return parser


def main(args: argparse.Namespace = None):
    if args is None:
        parser = RNNTParser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)
