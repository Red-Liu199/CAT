"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

import coreutils
import os
import argparse
import numpy as np
import model as model_zoo
import dataset as DataSet
from lm_train import LSTMPredictNet
from collections import OrderedDict
from typing import Union, Tuple, Sequence, Iterable, Literal
from warp_rnnt import rnnt_loss as RNNTLoss
from beam_search_base import BeamSearchRNNTransducer, BeamSearchConvTransducer, ConvMemBuffer

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def main(args: argparse.Namespace):
    if not torch.cuda.is_available():
        coreutils.highlight_msg("CPU only training is unsupported")
        return None

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"Global number of GPUs: {args.world_size}")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


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

    logger = OrderedDict({
        'log_train': ['epoch,loss,loss_real,net_lr,time'],
        'log_eval': ['loss_real,time']
    })
    manager = coreutils.Manager(logger, build_model, args)

    # get GPU info
    gpu_info = coreutils.gather_all_gpu_info(args.gpu)

    if args.rank == 0:
        print("> Model built.")
        print("  Model size:{:.2f}M".format(
            coreutils.count_parameters(manager.model)/1e6))

        coreutils.gen_readme(args.dir+'/readme.md',
                             model=manager.model, gpu_info=gpu_info)

    # training
    manager.run(train_sampler, trainloader, testloader, args)


class Transducer(nn.Module):
    def __init__(self, encoder: nn.Module = None, decoder: nn.Module = None, jointnet: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = jointnet

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        output_encoder, o_lens = self.encoder(inputs, input_lengths)
        padded_targets = torch.cat(
            [targets.new_zeros((targets.size(0), 1)), targets], dim=-1)
        output_decoder, _ = self.decoder(padded_targets)

        joint_out = self.joint(output_encoder, output_decoder)

        if isinstance(joint_out, tuple):
            joint_out = joint_out[0]

        ########## DEBUG CODE ###########
        # import matplotlib.pyplot as plt
        # harvest = torch.sum(torch.exp(joint_out[0, :o_lens[0], :, 1:].cpu().detach()), dim=-1).transpose(0,1).numpy()
        # print(harvest.shape)
        # fig, ax = plt.subplots()
        # im = ax.imshow(harvest)
        # plt.xlabel('T')
        # plt.ylabel('U')
        # plt.tight_layout()
        # plt.savefig("tmp.png", dpi=300)
        # plt.close()
        # exit(1)
        #################################

        loss = RNNTLoss(joint_out, targets.to(dtype=torch.int32), o_lens.to(device=joint_out.device,
                                                                            dtype=torch.int32), target_lengths.to(device=joint_out.device, dtype=torch.int32), reduction='mean', gather=True)

        return loss

    @torch.no_grad()
    def decode(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor, mode='beam', beam_size: int = 3) -> torch.LongTensor:
        encoder_output, o_lens = self.encoder(inputs, input_lengths)

        if mode == 'greedy':
            bos_token = inputs.new_zeros(1, 1).to(torch.long)
            decoder_output_init, hidden_dec_init = self.decoder(bos_token)
            outputs = []
            for seq, T in zip(encoder_output, o_lens):
                pred_tokens = []
                decoder_output = decoder_output_init
                hidden_dec = hidden_dec_init
                for t in range(T):
                    step_out = self.joint(seq[t], decoder_output.view(-1))
                    pred = step_out.argmax(dim=0)
                    pred = int(pred.item())
                    if pred != 0:
                        pred_tokens.append(pred)
                    decoder_input = torch.tensor(
                        [[pred]], device=inputs.device, dtype=torch.long)
                    decoder_output, hidden_dec = self.decoder(
                        decoder_input, hidden_dec)

                outputs.append(torch.tensor(
                    pred_tokens, device=inputs.device, dtype=torch.long))

            return coreutils.pad_list(outputs).to(torch.long)
        elif mode == 'beam':
            assert beam_size > 1
            searcher = BeamSearchRNNTransducer(self, beam_size, blank_id=0)
            return searcher.forward(encoder_output, o_lens.max())
        else:
            raise ValueError("Unknown decode mode: {}".format(mode))

    @torch.no_grad()
    def decode_conv(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor, beam_size: int, kernel_size: Union[int, Tuple[int, int]]):
        encoder_output, o_lens = self.encoder(inputs, input_lengths)
        searcher = BeamSearchConvTransducer(
            self, kernel_size, beam_size, blank_id=0).to(inputs.device)
        return searcher.forward(encoder_output, o_lens.max())


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

    def forward(self, encoder_output: torch.FloatTensor, decoder_output: torch.FloatTensor) -> torch.FloatTensor:
        assert (encoder_output.dim() == 3 and decoder_output.dim() == 3) or (
            encoder_output.dim() == 1 and decoder_output.dim() == 1)

        encoder_output = self.fc_enc(encoder_output)
        decoder_output = self.fc_dec(decoder_output)

        if encoder_output.dim() == 3:
            # expand the outputs
            T_max = encoder_output.size(1)
            U_max = decoder_output.size(1)

            encoder_output = encoder_output.unsqueeze(2)
            decoder_output = decoder_output.unsqueeze(1)

            encoder_output = encoder_output.repeat(
                [1, 1, U_max, 1])
            decoder_output = decoder_output.repeat(
                [1, T_max, 1, 1])

        if self._isHAT:
            conbined_input = encoder_output + decoder_output

            vocab_logits = self.fc(conbined_input)
            prob_blk = self.distr_blk(vocab_logits[:, :, :, :1])
            vocab_log_probs = torch.log(
                1-prob_blk)+torch.log_softmax(vocab_logits[:, :, :, 1:], dim=-1)
            outputs = torch.cat([torch.log(prob_blk), vocab_log_probs], dim=-1)
        else:
            outputs = self.fc(
                encoder_output+decoder_output).log_softmax(dim=-1)

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
def build_model(args, configuration: dict, dist: bool = True) -> Union[nn.Module, nn.parallel.DistributedDataParallel]:
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
            # FIXME: flexible decoder network like encoder.
            _model = LSTMPredictNet(**settings)

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

        if args.rank == 0:
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

    model = Transducer(encoder=encoder, decoder=decoder, jointnet=jointnet)

    if not all(not _model.freeze for _model in [encoder, decoder, jointnet]):
        setattr(model, 'requires_slice', True)

    if not dist:
        return model

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

    # FIXME: rm this dependencies
    setattr(args, 'iscrf', False)

    if not args.debug:
        ckptpath = os.path.join(args.dir, 'ckpt')
        if os.path.isdir(args.dir):
            os.makedirs(ckptpath, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"--dir={args.dir} is not a valid directory.")
    else:
        coreutils.highlight_msg("Debugging")
        # This is a hack, we won't read/write anything in debug mode.
        ckptpath = '/'

    setattr(args, 'ckptpath', ckptpath)
    if os.listdir(ckptpath) != [] and not args.debug and args.resume is None:
        raise FileExistsError(
            f"{args.ckptpath} is not empty! Refuse to run the experiment.")

    main(args)
