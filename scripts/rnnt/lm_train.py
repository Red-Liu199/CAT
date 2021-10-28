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
from data import BalancedDistributedSampler, CorpusDataset, sortedPadCollateLM

import time
import math
import argparse
from typing import Tuple, Union, List, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    coreutils.SetRandomSeed(args.seed)
    args.gpu = gpu
    torch.cuda.set_device(gpu)

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    test_set = CorpusDataset(args.devset)
    test_sampler = DistributedSampler(test_set)

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=sortedPadCollateLM())

    manager = coreutils.Manager(build_model, args, func_eval=evaluate)
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

    tr_set = CorpusDataset(args.trset)
    setattr(args, 'n_steps', 0)

    if args.databalance:
        coreutils.distprint(
            "> Enable data balanced loading, it takes a while to initialize...", args.gpu)
        train_sampler = BalancedDistributedSampler(
            tr_set, args.batch_size, args.len_norm)
        trainloader = DataLoader(
            tr_set, batch_sampler=train_sampler,
            num_workers=args.workers, pin_memory=True,
            collate_fn=sortedPadCollateLM())
        coreutils.distprint(
            "> Seq length info for balanced loading generated.", args.gpu)
        args.n_steps = train_sampler.total_size//args.batch_size//args.grad_accum_fold
    else:
        train_sampler = DistributedSampler(tr_set)

        trainloader = DataLoader(
            tr_set, batch_size=args.batch_size//ngpus_per_node, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler, collate_fn=sortedPadCollateLM())
        args.n_steps = len(trainloader)//args.grad_accum_fold

    # training
    manager.run(train_sampler, trainloader, testloader, args)


class AbsDecoder(nn.Module):
    """Abstract decoder class

    Args:
        num_classes (int): number of classes of tokens. a.k.a. the vocabulary size.
        n_emb (int): embedding hidden size.
        n_hid (int, optional): hidden size of decoder, also the dimension of input features of the classifier.
            if -1, will set `n_hid=n_emb`
        padding_idx (int, optional): index of padding lable, -1 to disable it.
        tied (bool, optional): flag of whether the embedding layer and the classifier layer share the weight. Default: False

    """

    def __init__(self, num_classes: int, n_emb: int, n_hid: int = -1, padding_idx: int = -1, tied: bool = False) -> None:
        super().__init__()
        if n_hid == -1:
            n_hid = n_emb

        assert n_emb > 0 and isinstance(
            n_emb, int), f"{self.__class__.__name__}: Invalid embedding size: {n_emb}"
        assert n_hid > 0 and isinstance(
            n_hid, int), f"{self.__class__.__name__}: Invalid hidden size: {n_hid}"
        assert (tied and (n_hid == n_emb)) or (
            not tied), f"{self.__class__.__name__}: tied=True is conflict with n_emb!=n_hid: {n_emb}!={n_hid}"
        assert padding_idx == -1 or (padding_idx > 0 and isinstance(padding_idx, -1) and padding_idx <
                                     num_classes), f"{self.__class__.__name__}: Invalid padding idx: {padding_idx}"

        if padding_idx == -1:
            self.embedding = nn.Embedding(num_classes, n_emb)
        else:
            self.embedding = nn.Embedding(
                num_classes, n_emb, padding_idx=padding_idx)

        self.classifier = nn.Linear(n_hid, num_classes)
        if tied:
            self.classifier.weight = self.embedding.weight


class LMTrainer(nn.Module):
    def __init__(self, lm: AbsDecoder = None):
        super().__init__()
        self.lm = lm    # type: AbsDecoder
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, *args, **kwargs) -> torch.FloatTensor:

        # preds: (N, S, C)
        preds, _ = self.lm(inputs, input_lengths=input_lengths)

        # squeeze preds by concat all sentences
        logits = []
        for i, l in enumerate(input_lengths):
            logits.append(preds[i, :l])

        # logits: (\sum{S_i}, C)
        logits = torch.cat(logits, dim=0)

        # targets: (\sum{S_i})
        loss = self.criterion(self.logsoftmax(logits), targets)
        return loss


class LSTMPredictNet(AbsDecoder):
    """
    RNN Decoder of Transducer
    Args:
        num_classes (int): number of classes, excluding the <blk>
        hdim (int): hidden state dimension of decoders
        norm (bool, optional): whether use layernorm
        variational_noise (tuple(float, float), optional): add variational noise with (mean, std)
        classical (bool, optional): whether use classical way of linear proj layer
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

    def __init__(self,
                 num_classes: int,
                 hdim: int,
                 norm: bool = False,
                 variational_noise: Union[Tuple[float,
                                                float], List[float]] = None,
                 classical: bool = True,
                 padding_idx: int = -1,
                 *rnn_args, **rnn_kwargs):
        super().__init__(num_classes, hdim, padding_idx=padding_idx)

        rnn_kwargs['batch_first'] = True
        if norm:
            self.norm = nn.LayerNorm([hdim])
        else:
            self.norm = None

        self.rnn = nn.LSTM(hdim, hdim, *rnn_args, **rnn_kwargs)
        if variational_noise is None:
            self._noise = None
        else:
            assert isinstance(variational_noise, tuple) or isinstance(
                variational_noise, list)
            variational_noise = [float(x) for x in variational_noise]
            assert variational_noise[1] > 0.

            self._mean_std = variational_noise
            self._noise = []  # type: List[Tuple[str, torch.nn.Parameter]]
            for name, param in self.rnn.named_parameters():
                if 'weight_' in name:
                    n_noise = name.replace("weight", "_noise")
                    self.register_buffer(n_noise, torch.empty_like(
                        param.data), persistent=False)
                    self._noise.append((n_noise, param))

        if classical:
            self.classifier = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(hdim, hdim)),
                ('act', nn.ReLU()),
                ('linear', nn.Linear(hdim, num_classes))
            ]))

    def forward(self, inputs: torch.LongTensor, hidden: torch.FloatTensor = None, input_lengths: torch.LongTensor = None) -> Tuple[torch.FloatTensor, Union[torch.FloatTensor, None]]:

        embedded = self.embedding(inputs)
        if self.norm is not None:
            embedded = self.norm(embedded)

        self.rnn.flatten_parameters()
        self.load_noise()
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
        self.unload_noise()

        out = self.classifier(rnn_out)

        return out, hidden_o

    def load_noise(self):
        if self._noise is None or not self.training:
            return

        for n_noise, param in self._noise:
            noise = getattr(self, n_noise)
            noise.normal_(*self._mean_std)
            param.data += noise

    def unload_noise(self):
        if self._noise is None or not self.training:
            return

        for n_noise, param in self._noise:
            noise = getattr(self, n_noise)
            param.data -= noise


class PlainPN(AbsDecoder):
    def __init__(self, num_classes: int, hdim: int, *args, **kwargs):
        super().__init__(num_classes, hdim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        embed_x = self.embedding(x)
        out = self.classifier(self.act(embed_x))
        return out, None


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(AbsDecoder):
    def __init__(self, num_classes: int, dim_hid: int, num_head: int, num_layers: int, dropout: float = 0.1, padding_idx: int = -1) -> None:
        super().__init__(num_classes, dim_hid, padding_idx=padding_idx)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(
            dim_hid, dropout=0.1, max_len=5000)

        encoder_layers = nn.TransformerEncoderLayer(
            dim_hid, num_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers)
        self.ninp = dim_hid

    def forward(self, src: torch.Tensor, input_lengths: Optional[torch.Tensor] = None, *args, **kwargs):
        # (N, S) -> (S, N)
        src = src.transpose(0, 1)
        T = src.size(0)
        if input_lengths is None:
            src_mask = None
            src_key_padding_mask = None
        else:
            src_mask = torch.triu(src.new_ones(
                T, T, dtype=torch.bool), diagonal=1)
            src_key_padding_mask = torch.arange(T, device=src.device)[
                None, :] >= input_lengths[:, None].to(src.device)

        embedded_src = self.embedding(src) * math.sqrt(self.ninp)
        embedded_src = self.pos_encoder(embedded_src)
        encoder_out = self.transformer_encoder(
            embedded_src, src_mask, src_key_padding_mask)
        output = self.classifier(encoder_out).transpose(0, 1)
        return output, None


@torch.no_grad()
def evaluate(testloader: DataLoader, args: argparse.Namespace, manager: coreutils.Manager):

    model = manager.model

    batch_time = coreutils.AverageMeter('Time', ':6.3f')
    losses = coreutils.AverageMeter('Loss', ':.3e')
    progress = coreutils.ProgressMeter(
        len(testloader),
        [batch_time, losses],
        prefix='Test: ')

    beg = time.time()
    cnt_batch = 0
    total_loss = 0.
    for i, minibatch in enumerate(testloader):
        logits, input_lengths, labels, label_lengths = minibatch
        logits, labels, input_lengths, label_lengths = logits.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        loss = model(logits, labels, input_lengths, label_lengths)
        batch_sum_loss = loss * logits.size(0)  # type: torch.Tensor
        n_batch = batch_sum_loss.new_tensor(logits.size(0), dtype=torch.long)

        dist.all_reduce(batch_sum_loss, dist.ReduceOp.SUM)
        dist.all_reduce(n_batch, dist.ReduceOp.SUM)

        # measure accuracy and record loss
        losses.update((batch_sum_loss/n_batch).item())
        cnt_batch += n_batch.item()
        total_loss += batch_sum_loss.item()

        if ((i+1) % args.print_freq == 0 or args.debug) and args.gpu == 0:
            progress.display(i+1)

    avgloss = total_loss / cnt_batch
    manager.log_update(
        [avgloss, time.time() - beg], loc='log_eval')

    # use ppl as evalution metric
    return math.exp(avgloss)


def build_model(args, configuration, dist=True, wrapper=True) -> LMTrainer:
    def _build_decoder(config) -> nn.Module:
        LMNet = eval(config['type'])    # type: Union[PlainPN | LSTMPredictNet]
        NetKwargs = config['kwargs']
        return LMNet(**NetKwargs)

    assert 'decoder' in configuration

    decoder = _build_decoder(configuration['decoder'])

    if wrapper:
        model = LMTrainer(decoder)
    else:
        model = decoder

    if not dist:
        return model

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    return model


if __name__ == "__main__":
    parser = coreutils.BasicDDPParser()

    args = parser.parse_args()

    setPath(args)

    main_spawner(args, main_worker)
