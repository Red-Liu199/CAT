"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

import coreutils
import model as model_zoo
import lm_train as pn_zoo
import dataset as DataSet
from am_train import setPath, main_spawner
from beam_search_base import ConvMemBuffer
from _layers import TimeReduction

import os
import argparse
from collections import OrderedDict
from gather import gathersum, gathercat
from warp_rnnt import rnnt_loss as RNNTLoss
from typing import Union, Tuple, Sequence, Iterable, Literal, List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    coreutils.SetRandomSeed(args.seed)
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.batch_size = args.batch_size // ngpus_per_node

    if args.h5py:
        data_format = "hdf5"
        coreutils.highlight_msg(
            "H5py reading might cause error with Multi-GPUs")
        Dataset = DataSet.SpeechDataset
        if args.trset is None or args.devset is None:
            raise FileNotFoundError(
                "With '--hdf5' option, you must specify data location with '--trset' and '--devset'.")
    else:
        data_format = "pickle"
        Dataset = DataSet.SpeechDatasetPickle

    if args.trset is None:
        args.trset = os.path.join(args.data, f'{data_format}/tr.{data_format}')
    if args.devset is None:
        args.devset = os.path.join(
            args.data, f'{data_format}/cv.{data_format}')

    tr_set = Dataset(args.trset)
    test_set = Dataset(args.devset)

    train_sampler = DistributedSampler(tr_set)
    test_sampler = DistributedSampler(test_set)
    test_sampler.set_epoch(1)

    trainloader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, collate_fn=DataSet.sortedPadCollateTransducer())

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=DataSet.sortedPadCollateTransducer())

    manager = coreutils.Manager(build_model, args)

    # get GPU info
    gpu_info = coreutils.gather_all_gpu_info(args.gpu)

    coreutils.distprint("> Model built.", args.gpu)
    coreutils.distprint("  Model size:{:.2f}M".format(
        coreutils.count_parameters(manager.model)/1e6), args.gpu)
    if args.rank == 0 and not args.debug:
        coreutils.gen_readme(args.dir+'/readme.md',
                             model=manager.model, gpu_info=gpu_info)

    # training
    manager.run(train_sampler, trainloader, testloader, args)


class PackedSequence():
    def __init__(self, xs: Union[Sequence[torch.Tensor], torch.Tensor] = None, xn: torch.LongTensor = None) -> None:
        if xs is None:
            return

        if xn is not None:
            assert isinstance(xs, torch.Tensor)
            if xs.dim() == 3:
                V = xs.size(-1)
            elif xs.dim() == 2:
                xs = xs.unsqueeze(2)
                V = 1
            else:
                raise NotImplementedError

            if xs.dtype != torch.float:
                # this might be slow
                self._data = torch.cat([xs[i, :xn[i]].view(-1, V)
                                       for i in range(xn.size(0))], dim=0)
            else:
                self._data = gathercat(xs, xn)
            self._lens = xn
        else:
            # identical to torch.nn.utils.rnn.pad_sequence
            assert all(x.size()[1:] == xs[0].size()[1:] for x in xs)

            _lens = [x.size(0) for x in xs]
            self._data = xs[0].new_empty(
                ((sum(_lens),)+xs[0].size()[1:]))
            pref = 0
            for x, l in zip(xs, _lens):
                self._data[pref:pref+l] = x
                pref += l
            self._lens = torch.LongTensor(_lens)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def batch_sizes(self) -> torch.LongTensor:
        return self._lens

    def to(self, device):
        newpack = PackedSequence()
        newpack._data = self._data.to(device)
        newpack._lens = self._lens.to(device)
        return newpack

    def set(self, data: torch.Tensor):
        assert data.size(0) == self._data.size(0)
        self._data = data
        return self

    def unpack(self) -> Tuple[List[torch.Tensor], torch.LongTensor]:
        out = []

        pref = 0
        for l in self._lens:
            out.append(self._data[pref:pref+l])
            pref += l
        return out, self._lens

    def __add__(self, _y) -> torch.Tensor:
        return gathersum(self._data, _y._data, self._lens, _y._lens)


class Transducer(nn.Module):
    def __init__(self,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 jointnet: nn.Module = None,
                 compact: bool = False,
                 time_reduction: int = 1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = jointnet
        self._compact = compact and isinstance(jointnet, JointNet)
        assert isinstance(time_reduction, int) and time_reduction >= 1
        if time_reduction == 1:
            self._t_reduction = None
        else:
            self._t_reduction = TimeReduction(time_reduction)

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        output_encoder, o_lens = self.encoder(inputs, input_lengths)
        # introduce time reduction layer
        if self._t_reduction is not None:
            output_encoder, o_lens = self._t_reduction(output_encoder, o_lens)

        padded_targets = torch.cat(
            [targets.new_zeros((targets.size(0), 1)), targets], dim=-1)
        output_decoder, _ = self.decoder(padded_targets)
        if self._compact:
            packed_enc = PackedSequence(output_encoder, o_lens)
            packed_dec = PackedSequence(output_decoder, target_lengths+1)
            joint_out = self.joint(packed_enc, packed_dec)
            targets = PackedSequence(targets, target_lengths).data
        else:
            joint_out = self.joint(output_encoder, output_decoder)

        if isinstance(joint_out, tuple):
            joint_out = joint_out[0]

        loss = RNNTLoss(joint_out, targets.to(dtype=torch.int32), o_lens.to(device=joint_out.device,
                        dtype=torch.int32), target_lengths.to(device=joint_out.device, dtype=torch.int32),
                        reduction='mean', gather=True, compact=self._compact)

        return loss


class JointNet(nn.Module):
    """
    Joint `encoder_output` and `decoder_output`.
    Args:
        encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size ``(batch, time_steps, dimensionA)``
        decoder_output (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size ``(batch, label_length, dimensionB)``
    Returns:
        outputs (torch.FloatTensor): outputs of joint `encoder_output` and `decoder_output`. `FloatTensor` of size ``(batch, time_steps, label_length, dimensionA + dimensionB)``
    """

    def __init__(self, odim_encoder: int, odim_decoder: int, num_classes: int, HAT: bool = False, act: Union[Literal['tanh'], Literal['relu']] = 'tanh'):
        super().__init__()
        in_features = odim_encoder+odim_decoder
        self.fc_enc = nn.Linear(odim_encoder, in_features)
        self.fc_dec = nn.Linear(odim_decoder, in_features)
        self._isHAT = HAT
        if act == 'tanh':
            act_layer = nn.Tanh()
        elif act == 'relu':
            act_layer = nn.ReLU()
        else:
            raise NotImplementedError(f"Unknown activation layer type: {act}")

        self.fc = nn.Sequential(
            act_layer,
            nn.Linear(in_features, num_classes)
        )
        if HAT:
            """
            Implementation of Hybrid Autoregressive Transducer (HAT)
            https://arxiv.org/abs/2003.07705
            """
            self.distr_blk = nn.Sigmoid()

    def forward(self, encoder_output: Union[torch.Tensor, PackedSequence], decoder_output: Union[torch.Tensor, PackedSequence]) -> torch.FloatTensor:

        if isinstance(encoder_output, PackedSequence) and isinstance(decoder_output, PackedSequence):
            # compact memory mode, gather the tensors without padding
            packed_encoder_output = encoder_output.set(
                self.fc_enc(encoder_output.data))    # type:torch.Tensor
            packed_decoder_output = decoder_output.set(
                self.fc_dec(decoder_output.data))

            # shape: (\sum_{Ti(Ui+1)}, V)
            expanded_out = packed_encoder_output + packed_decoder_output
        elif isinstance(encoder_output, torch.Tensor) and isinstance(decoder_output, torch.Tensor):

            assert (encoder_output.dim() == 3 and decoder_output.dim() == 3) or (
                encoder_output.dim() == 1 and decoder_output.dim() == 1)

            encoder_output = self.fc_enc(encoder_output)
            decoder_output = self.fc_dec(decoder_output)

            if encoder_output.dim() == 3:
                encoder_output = encoder_output.unsqueeze(2)
                decoder_output = decoder_output.unsqueeze(1)

            expanded_out = encoder_output + decoder_output
        else:
            raise NotImplementedError(
                "Output of encoder and decoder being fed into jointnet should be of same type. Expect (Tensor, Tensor) or (PackedSequence, PackedSequence), instead ({}, {})".format(type(encoder_output), type(decoder_output)))

        if self._isHAT:
            vocab_logits = self.fc(expanded_out)
            prob_blk = self.distr_blk(vocab_logits[..., :1])
            vocab_log_probs = torch.log(
                1-prob_blk)+torch.log_softmax(vocab_logits[..., 1:], dim=-1)
            outputs = torch.cat([torch.log(prob_blk), vocab_log_probs], dim=-1)
        else:
            outputs = self.fc(expanded_out).log_softmax(dim=-1)

        return outputs


class CausalConv2d(nn.Module):
    """
    Causal 2d-conv. Applied to (N, T, U, K) dim tensors.

    Args:
        in_channels  (int): the input dimension (namely K)
        out_channels (int): the output dimension
        kernel_size  (int): the kernel size (kernel is square)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], islast: bool = False):
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"Invalid initialization for CausalConv2d: {in_channels}, {out_channels}, {kernel_size}")

        if islast:
            self.causal_conv = nn.Sequential(OrderedDict({
                # seperate convlution
                'depth_conv': nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels),
                'point_conv': nn.Conv2d(in_channels, out_channels, kernel_size=1)
                # 'conv': nn.Conv2d(in_channels, out_channels, kernel_size)
            }))
        else:
            # FIXME: I think a normalization is helpful so that the padding won't change the distribution of features.
            self.causal_conv = nn.Sequential(OrderedDict({
                # seperate convlution
                'depth_conv': nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels),
                'point_conv': nn.Conv2d(in_channels, out_channels, kernel_size=1),
                'relu': nn.ReLU(inplace=True),
                'bn': nn.BatchNorm2d(in_channels),
                # 'conv': nn.Conv2d(in_channels, out_channels, kernel_size)
            }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.causal_conv(x)


class ConvJointNet(nn.Module):
    def __init__(self, odim_encoder: int, odim_decoder: int, num_classes: int, kernel_size: Union[int, Tuple[int, int]] = (3, 3)):
        super().__init__()
        K = max(odim_encoder, odim_decoder)
        self.fc_enc = nn.Linear(odim_encoder, K)
        self.fc_dec = nn.Linear(odim_decoder, K)
        self.act = nn.Tanh()

        # kernel among (T, U)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = [kernel_size[0] - 1, kernel_size[1] - 1]
        '''
        padding (int, tuple(int, int)): padding to the top of T and left of U,
            which would be extended to (padding[0]+T, padding[1]+U)
        '''
        self.padding = nn.ConstantPad2d(
            padding=(padding[1], 0, padding[0], 0), value=0.)
        self.conv = CausalConv2d(K, num_classes, kernel_size, islast=True)

    def forward(self, encoder_output: torch.FloatTensor, decoder_output: torch.FloatTensor, buffers: Sequence[ConvMemBuffer] = None, t: int = -1, u: int = -1) -> Tuple[torch.Tensor, Union[Sequence[ConvMemBuffer], None]]:

        if encoder_output.dim() == 3:
            encoder_output = self.fc_enc(encoder_output)
            decoder_output = self.fc_dec(decoder_output)

            # training, expand the outputs

            # (N, T_max, K) -> (N, K, T_max)
            encoder_output = encoder_output.transpose(1, 2).contiguous()
            # (N, U_max, K) -> (N, K, U_max)
            decoder_output = decoder_output.transpose(1, 2).contiguous()
            # (N, K, T_max) -> (N, K, T_max, 1)
            encoder_output = encoder_output.unsqueeze(3)
            # (N, K, U_max) -> (N, K, 1, U_max)
            decoder_output = decoder_output.unsqueeze(2)
            # (N, K, T_max, U_max)
            expanded_x = self.act(encoder_output + decoder_output)
            # (N, K, T_max, U_max) -> (N, T_max, U_max, V) -> (N, T_max, U_max, V)
            padded_x = self.padding(expanded_x)
            conv_x = self.conv(padded_x).permute(
                0, 2, 3, 1).contiguous()  # type: torch.Tensor

            return conv_x.log_softmax(dim=-1), None

        else:
            # decoding
            buffers = [x.replica() for x in buffers]
            buffers[0].append(t, u, encoder_output, decoder_output)
            encoder_output, decoder_output = buffers[0].mem

            encoder_output = self.fc_enc(encoder_output)
            decoder_output = self.fc_dec(decoder_output)

            # (S_t, K) -> (K, S_t)
            encoder_output = encoder_output.transpose(0, 1).contiguous()
            # (S_u, K) -> (K, S_u)
            decoder_output = decoder_output.transpose(0, 1).contiguous()
            # (K, S_t) -> (K, S_t, 1)
            encoder_output = encoder_output.unsqueeze(2)
            # (K, S_u) -> (K, 1, S_u)
            decoder_output = decoder_output.unsqueeze(1)

            # (K, S_t, S_u)
            expanded_x = self.act(encoder_output + decoder_output)

            # (K, S_t, S_u) -> (1, K, S_t, S_u) -> (1, V, 1, 1)
            conv_x = self.conv(expanded_x.unsqueeze(0))

            return conv_x.view(-1).log_softmax(dim=-1), buffers


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
            _model = getattr(model_zoo, config['type'])(
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
                # type: Union[JointNet, ConvJointNet]
                AbsNet = eval(config['type'])
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
                coreutils.highlight_msg(
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

        if args.rank == 0 and verbose:
            if 'pretrained' not in config:
                _path = ''
            else:
                _path = config['pretrained']
            print("{}: freeze={} | pretrained at {}".format(
                module, _model.freeze, _path))
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

    # for capability of old settings
    if 'transducer' in configuration:
        transducer_kwargs = configuration["transducer"]     # type: dict
    else:
        transducer_kwargs = {}

    if 'compact' not in transducer_kwargs:
        transducer_kwargs['compact'] = False

    model = Transducer(encoder=encoder, decoder=decoder,
                       jointnet=jointnet, **transducer_kwargs)

    if not all(not _model.freeze for _model in [encoder, decoder, jointnet]):
        setattr(model, 'requires_slice', True)

    if not dist:
        return model

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


if __name__ == "__main__":
    parser = coreutils.BasicDDPParser()
    parser.add_argument("--h5py", action="store_true",
                        help="Load data with H5py, defaultly use pickle (recommended).")

    args = parser.parse_args()

    setPath(args)

    main_spawner(args, main_worker)
