# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Decoder module impl

"""
import math
from collections import OrderedDict
from typing import Union, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import GPT2Model, GPT2Config


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


'''
Reference:
https://github.com/pytorch/examples/blob/3970e068c7f18d2d54db2afee6ddd81ef3f93c24/word_language_model/model.py#L108
'''

# FIXME (huahuan): deprecate this once the huggingface one is tested.
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

    def forward(self, src: torch.Tensor, cache: torch.Tensor = None, input_lengths: Optional[torch.Tensor] = None, *args, **kwargs):
        # (N, S) -> (S, N)
        src = src.transpose(0, 1)
        # this cache way is not efficient at all, but quite simple
        if 'hidden' in kwargs and cache is None:
            cache = kwargs['hidden']
        if cache is not None:
            # FIXME (huahuan): deal with paddings for N > 1
            src = torch.cat([cache, src], dim=1)
            cache = src.detach()

        T = src.size(0)
        src_mask = torch.triu(src.new_ones(
            T, T, dtype=torch.bool), diagonal=1)
        if input_lengths is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = torch.arange(T, device=src.device)[
                None, :] >= input_lengths[:, None].to(src.device)

        embedded_src = self.embedding(src) * math.sqrt(self.ninp)
        embedded_src = self.pos_encoder(embedded_src)
        encoder_out = self.transformer_encoder(
            embedded_src, src_mask, src_key_padding_mask)
        output = self.classifier(encoder_out).transpose(0, 1)
        return output, cache


class CausalTransformer(AbsDecoder):
    def __init__(self, num_classes: int, dim_hid: int, num_head: int, num_layers: int, attn_dropout: float = 0.1, padding_idx: int = -1) -> None:
        super().__init__(num_classes, dim_hid, padding_idx=padding_idx)
        configuration = GPT2Config(
            vocab_size=num_classes, n_embd=dim_hid,
            n_layer=num_layers, n_head=num_head, attn_pdrop=attn_dropout)
        self.trans = GPT2Model(configuration)
        # use my own token embedding layer
        self.trans.wte = None

    def forward(self, src_ids: torch.Tensor, cache: torch.Tensor = None, input_lengths: Optional[torch.Tensor] = None, *args, **kwargs):
        # (N, S) -> (N, S, D])
        embed_x = self.embedding(src_ids)
        use_cache = not self.training

        if input_lengths is None:
            padding_mask = None
        else:
            # 1 for not masked, 0 for masked,
            # this behavior is different from PyTorch nn.Transformer
            padding_mask = torch.arange(src_ids.size(1), device=src_ids.device)[
                None, :] < input_lengths[:, None].to(src_ids.device)
            padding_mask = padding_mask.to(torch.float)

        if 'hidden' in kwargs and cache is None:
            cache = kwargs['hidden']

        clm_out = self.trans(inputs_embeds=embed_x,
                             attention_mask=padding_mask,
                             past_key_values=cache, use_cache=use_cache)
        logits = self.classifier(clm_out['last_hidden_state'])
        if use_cache:
            return logits, clm_out['past_key_values']
        else:
            return logits, None
