"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

import coreutils
from am_train import setPath, main_spawner
from dataset import sortedPadCollateLM, CorpusDataset

import os
import argparse
from typing import Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    coreutils.SetRandomSeed(args.seed)
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.batch_size = args.batch_size // ngpus_per_node

    test_set = CorpusDataset(args.devset)
    tr_set = CorpusDataset(args.trset)

    train_sampler = DistributedSampler(tr_set)
    test_sampler = DistributedSampler(test_set)
    test_sampler.set_epoch(1)

    trainloader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, collate_fn=sortedPadCollateLM())

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=sortedPadCollateLM())

    manager = coreutils.Manager(build_model, args)
    # lm training does not need specaug
    manager.specaug = None

    # get GPU info
    gpu_info = coreutils.gather_all_gpu_info(args.gpu)

    coreutils.distprint("> Model built.", args.gpu)
    coreutils.distprint("  Model size:{:.2f}M".format(
        coreutils.count_parameters(manager.model)/1e6), args.gpu)
    if args.rank == 0 and not args.debug:
        coreutils.gen_readme(args.dir+'/readme.md',
                             model=manager.model, gpu_info=gpu_info)

    if args.evaluate:
        manager.model.eval()

        ppl = 0.
        for minibatch in testloader:
            logits, input_lengths, labels, label_lengths, _ = minibatch
            logits, labels, input_lengths, label_lengths = logits.cuda(
                args.gpu, non_blocking=True), labels, input_lengths, label_lengths

            ppl_i = manager.model.module.test(logits, labels, input_lengths)
            dist.all_reduce(ppl_i)
            ppl += ppl_i

        coreutils.distprint("PPL for {} sentences: {:.2f}".format(
            len(test_set), ppl/len(test_set)), args.gpu)
        return

    # training
    manager.run(train_sampler, trainloader, testloader, args)


class LMTrainer(nn.Module):
    def __init__(self, lm: nn.Module = None):
        super().__init__()
        self.lm = lm    # type: LSTMPredictNet
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def test(self, inputs: torch.LongTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor):
        targets = targets.to(inputs.device)
        # preds: (N, S, C)
        preds, _ = self.lm(inputs)
        log_p = torch.log_softmax(preds, dim=-1)
        # squeeze log_p by concat all sentences
        # log_p: (\sum{S_i}, C)
        log_p = torch.cat([log_p[i, :l]
                          for i, l in enumerate(input_lengths)], dim=0)

        # target_mask: (\sum{S_i}, C)
        target_mask = torch.arange(log_p.size(1), device=log_p.device)[
            None, :] == targets[:, None]
        # log_p: (\sum{S_i}, )
        log_p = torch.sum(log_p * target_mask, dim=-1)

        ppl = [-1./input_lengths[i]*torch.sum(log_p[input_lengths[:i].sum(
        ):input_lengths[:(i+1)].sum()]) for i in range(input_lengths.size(0))]
        ppl = torch.stack(ppl)

        return torch.sum(torch.exp(ppl))

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        # preds: (N, S, C)
        preds, _ = self.lm(inputs)

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
        *rnn_args/**rnn_kwargs : any arguments that can be passed as 
            nn.LSTM(*rnn_args, **rnn_kwargs)
    Inputs: inputs, hidden_states, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
    Returns:
        (Tensor, Tensor):
        * decoder_outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """

    def __init__(self, num_classes: int, hdim: int, norm: bool = False, *rnn_args, **rnn_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hdim)

        rnn_kwargs['batch_first'] = True
        if norm:
            self.norm = nn.LayerNorm([hdim])
        else:
            self.norm = None

        self.rnn = nn.LSTM(hdim, hdim, *rnn_args, **rnn_kwargs)
        if 'bidirectional' in rnn_kwargs and rnn_kwargs['bidirectional']:
            odim = hdim*2
        else:
            odim = hdim
        self.out_proj = nn.Linear(odim, odim)

        self.classifier = nn.Sequential(OrderedDict({
            'act': nn.ReLU(),
            'linear': nn.Linear(odim, num_classes)
        }))

    def forward(self, inputs: torch.LongTensor, hidden: torch.FloatTensor = None, input_lengths: torch.LongTensor = None) -> Tuple[torch.FloatTensor, Union[torch.FloatTensor, None]]:

        embedded = self.embedding(inputs)
        if self.norm is not None:
            embedded = self.norm(embedded)

        self.rnn.flatten_parameters()
        '''
        since the batch is sorted by time_steps length rather the target length
        ...so here we don't use the pack_padded_sequence()
        '''
        if input_lengths is not None:
            packed_input = pack_padded_sequence(
                embedded, input_lengths.to("cpu"), batch_first=True)
            packed_output, hidden_o = self.rnn(packed_input, hidden)
            rnn_out, olens = pad_packed_sequence(
                packed_output, batch_first=True)
        else:
            rnn_out, hidden_o = self.rnn(embedded, hidden)

        proj_out = self.out_proj(rnn_out)
        out = self.classifier(proj_out)

        return out, hidden_o


class PlainPN(nn.Module):
    def __init__(self, num_classes: int, hdim: int, *args, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hdim)

    def forward(self, x: torch.Tensor, *args):
        return self.embedding(x), None


def build_model(args, configuration, dist=True, wrapper=True) -> LMTrainer:
    def _build_decoder(config) -> nn.Module:
        LMNet = eval(config['type'])    # type: Union[PlainPN | LSTMPredictNet]
        NetKwargs = config['kwargs']
        # FIXME: flexible decoder network like encoder.
        return LMNet(**NetKwargs)

    assert 'decoder' in configuration

    decoder = _build_decoder(configuration['decoder'])

    if wrapper:
        model = LMTrainer(decoder)
    else:
        model = decoder

    if not dist:
        return model

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


if __name__ == "__main__":
    parser = coreutils.BasicDDPParser()
    parser.add_argument("--evaluate", action="store_true", default=False)

    args = parser.parse_args()

    setPath(args)

    main_spawner(args, main_worker)
