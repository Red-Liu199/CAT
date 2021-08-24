"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

import utils
import os
import argparse
import numpy as np
import model as model_zoo
import dataset as DataSet
from _specaug import SpecAug
from collections import OrderedDict
from typing import Union, Tuple, Sequence
from warp_rnnt import rnnt_loss as RNNTLoss
from beam_search_base import BeamSearchRNNTransducer

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def main(args):
    if not torch.cuda.is_available():
        utils.highlight_msg("CPU only training is unsupported.")
        return None

    os.makedirs(args.dir+'/ckpt', exist_ok=True)
    setattr(args, 'iscrf', False)
    setattr(args, 'ckptpath', args.dir+'/ckpt')
    if os.listdir(args.ckptpath) != [] and not args.debug and args.resume is None:
        raise FileExistsError(
            f"{args.ckptpath} is not empty! Refuse to run the experiment.")

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"Global number of GPUs: {args.world_size}")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.batch_size = args.batch_size // ngpus_per_node

    if args.h5py:
        data_format = "hdf5"
        utils.highlight_msg("H5py reading might cause error with Multi-GPUs.")
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
    manager = utils.Manager(logger, build_model, args)

    # get GPU info
    gpu_info = utils.gather_all_gpu_info(args.gpu)

    if args.rank == 0:
        print("> Model built.")
        print("  Model size:{:.2f}M".format(
            utils.count_parameters(manager.model)/1e6))

        utils.gen_readme(args.dir+'/readme.md',
                         model=manager.model, gpu_info=gpu_info)

    # training
    manager.run(train_sampler, trainloader, testloader, args)


class Transducer(nn.Module):
    def __init__(self, lm: nn.Module = None):
        super().__init__()
        self.lm = lm    # type:nn.Module
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:
        
        # preds: (N, S, C)
        preds, _ = self.lm(inputs, input_lengths)

        # squeeze preds by concat all sentences
        logits = []
        for i, l in enumerate(input_lengths):
            logits.append(preds[i, :l])
        
        # logits: (\sum{S_i}, C)
        logits = torch.cat(logits, dim=0)
        # targets: (\sum{S_i})
        loss = self.criterion(logits, targets)

        return loss


class LSTMPredictNet(nn.Module):
    """
    RNN Decoder of Transducer
    Args:
        num_classes (int): number of classes, excluding the <blk>
        hdim (int): hidden state dimension of decoders
        odim (int): output dimension of decoder
        *rnn_args/**rnn_kwargs : any arguments that can be passed as 
            nn.LSTM(*rnn_args, **rnn_kwargs)
    Inputs: inputs, input_lengths, hidden_states
        inputs (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
    Returns:
        (Tensor, Tensor):
        * decoder_outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    Reference:
        A Graves: Sequence Transduction with Recurrent Neural Networks
        https://arxiv.org/abs/1211.3711.pdf
    """

    def __init__(self, num_classes: int, hdim: int, odim: int, *rnn_args, **rnn_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hdim)

        rnn_kwargs['batch_first'] = True
        self.rnn = nn.LSTM(hdim, hdim, *rnn_args, **rnn_kwargs)
        if 'bidirectional' in rnn_kwargs and rnn_kwargs['bidirectional']:
            self.out_proj = nn.Linear(hdim*2, odim)
        else:
            self.out_proj = nn.Linear(hdim, odim)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(odim, num_classes)
        )

    def forward(self, input: torch.LongTensor, hidden: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, Union[torch.FloatTensor, None]]:

        embedded = self.embedding(input)
        self.rnn.flatten_parameters()
        '''
        since the batch is sorted by time_steps length rather the target length
        ...so here we don't use the pack_padded_sequence()
        '''
        # packed_input = pack_padded_sequence(
        #     embedded, input_lengths.to("cpu"), batch_first=True)
        # packed_output, hidden_o = self.rnn(packed_input, hidden)
        # rnn_out, olens = pad_packed_sequence(packed_output, batch_first=True)
        rnn_out, hidden_o = self.rnn(embedded, hidden)

        out = self.out_proj(rnn_out)
        out = self.classifier(out)

        return out, hidden_o


def build_model(args, configuration, dist=True) -> nn.Module:
    def _build_encoder(config) -> nn.Module:
        NetKwargs = config['kwargs']
        _encoder = getattr(model_zoo, config['type'])(**NetKwargs)

        # FIXME: this is a hack
        _encoder.classifier = nn.Identity()

        return _encoder

    def _build_decoder(config) -> nn.Module:
        NetKwargs = config['kwargs']
        # FIXME: flexible decoder network like encoder.
        return LSTMPredictNet(**NetKwargs)

    def _build_jointnet(config) -> nn.Module:
        """
            The joint network accept the concatence of outputs of the 
            encoder and decoder. So the input dimensions MUST match that.
        """
        NetKwargs = config['kwargs']
        return JointNet(**NetKwargs)

    assert 'encoder' in configuration and 'decoder' in configuration and 'joint' in configuration

    encoder = _build_encoder(configuration['encoder'])
    decoder = _build_decoder(configuration['decoder'])
    jointnet = _build_jointnet(configuration['joint'])

    model = Transducer(encoder=encoder, decoder=decoder,
                       jointnet=jointnet)
    if not dist:
        return model

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    if hasattr(args, "pretrained_encoder") and args.pretrained_encoder is not None:

        assert os.path.isfile(args.pretrained_encoder)
        checkpoint = torch.load(args.pretrained_encoder,
                                map_location=f'cuda:{args.gpu}')

        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            # replace the 'infer' with 'encoder'
            new_state_dict[k.replace('infer', 'encoder')] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recognition argument")

    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Distributed Data Parallel')

    parser.add_argument("--seed", type=int, default=0,
                        help="Manual seed.")
    parser.add_argument("--grad-accum-fold", type=int, default=1,
                        help="Utilize gradient accumulation for K times. Default: K=1")
    parser.add_argument("--pretrained-encoder", type=str, default=None,
                        help="Path to pretrained encoder model")

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument("--decode", action="store_true",
                        help="Configure to debug settings, would overwrite most of the options.")

    parser.add_argument("--debug", action="store_true",
                        help="Configure to debug settings, would overwrite most of the options.")
    parser.add_argument("--h5py", action="store_true",
                        help="Load data with H5py, defaultly use pickle (recommended).")

    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of training procedure.")

    parser.add_argument("--data", type=str, default=None,
                        help="Location of training/testing data.")
    parser.add_argument("--trset", type=str, default=None,
                        help="Location of training data. Default: <data>/[pickle|hdf5]/tr.[pickle|hdf5]")
    parser.add_argument("--devset", type=str, default=None,
                        help="Location of dev data. Default: <data>/[pickle|hdf5]/cv.[pickle|hdf5]")
    parser.add_argument("--dir", type=str, default=None, metavar='PATH',
                        help="Directory to save the log and model files.")

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:13943', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    if args.debug:
        utils.highlight_msg("Debugging.")

    main(args)
