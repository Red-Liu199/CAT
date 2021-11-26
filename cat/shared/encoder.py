# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Keyu An,
#         Zheng Huahuan (maxwellzh@outlook.com)

"""Decoder modules impl
"""
from . import layer as c_layers

import numpy as np
from collections import OrderedDict
from typing import Literal

import math
import torch
import torch.nn as nn


def get_vgg2l_odim(idim, in_channel=1, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


class AbsEncoder(nn.Module):
    def __init__(self, with_head: bool = True, num_classes: int = -1, n_hid: int = -1) -> None:
        super().__init__()
        if with_head:
            assert num_classes > 0, f"Vocab size should be > 0, instead {num_classes}"
            assert n_hid > 0, f"Hidden size should be > 0, instead {n_hid}"
            self.classifier = nn.Linear(n_hid, num_classes)
        else:
            self.classifier = nn.Identity()

    def impl_forward(self, *args, **kwargs):
        '''Implement the forward funcion w/o classifier'''
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        out = self.impl_forward(*args, **kwargs)
        if isinstance(out, tuple):
            _co = self.classifier(out[0])
            return (_co,)+out[1:]
        else:
            return self.classifier(out)


class LSTM(AbsEncoder):
    def __init__(self,
                 idim: int,
                 hdim: int,
                 n_layers: int,
                 num_classes: int,
                 dropout: float,
                 with_head: bool = True,
                 bidirectional: bool = False):
        super().__init__(with_head=with_head, num_classes=num_classes,
                         n_hid=(2*hdim if bidirectional else hdim))

        self.lstm = c_layers._LSTM(
            idim, hdim, n_layers, dropout, bidirectional=bidirectional)

    def impl_forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        return self.lstm(x, ilens, hidden)


class VGGLSTM(LSTM):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int, dropout: float, with_head: bool = True, in_channel: int = 3, bidirectional: int = False):
        super().__init__(get_vgg2l_odim(idim, in_channel),
                         hdim, n_layers, num_classes, dropout, with_head, bidirectional)

        self.VGG = c_layers.VGG2L(in_channel)

    def impl_forward(self, x: torch.Tensor, ilens: torch.Tensor):
        vgg_o, vgg_lens = self.VGG(x, ilens)
        return super().impl_forward(vgg_o, vgg_lens)


class LSTMrowCONV(AbsEncoder):
    def __init__(self, idim: int, hdim: int, n_layers: int, dropout: float, with_head: bool = True,  num_classes: int = -1) -> None:
        super().__init__(with_head=with_head, num_classes=num_classes, n_hid=hdim)

        self.lstm = c_layers._LSTM(idim, hdim, n_layers, dropout)
        self.lookahead = c_layers.Lookahead(hdim, context=5)

    def impl_forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        lstm_out, olens = self.lstm(x, ilens, hidden)
        ahead_out = self.lookahead(lstm_out)
        return ahead_out, olens


class TDNN_NAS(AbsEncoder):
    def __init__(self, idim: int, hdim: int, dropout: float = 0.5, num_classes: int = -1, with_head: bool = True) -> None:
        super().__init__(with_head=with_head, num_classes=num_classes, n_hid=hdim)

        self.dropout = nn.Dropout(dropout)
        self.tdnns = nn.ModuleDict(OrderedDict([
            ('tdnn0', c_layers.TDNN(idim, hdim, half_context=2, dilation=1)),
            ('tdnn1', c_layers.TDNN(idim, hdim, half_context=2, dilation=2)),
            ('tdnn2', c_layers.TDNN(idim, hdim, half_context=2, dilation=1)),
            ('tdnn3', c_layers.TDNN(idim, hdim, stride=3)),
            ('tdnn4', c_layers.TDNN(idim, hdim, half_context=2, dilation=2)),
            ('tdnn5', c_layers.TDNN(idim, hdim, half_context=2, dilation=1)),
            ('tdnn6', c_layers.TDNN(idim, hdim, half_context=2, dilation=2))
        ]))

    def impl_forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tmp_x, tmp_lens = x, ilens
        for i, tdnn in enumerate(self.tdnns.values):
            if i < len(self.tdnns)-1:
                tmp_x = self.dropout(x)
            tmp_x, tmp_lens = tdnn(tmp_x, tmp_lens)

        return tmp_x, tmp_lens


class TDNN_LSTM(AbsEncoder):
    def __init__(self, idim: int, hdim: int, n_layers: int, dropout: float, num_classes: int = -1, with_head: bool = True) -> None:
        super().__init__(with_head=with_head, num_classes=num_classes, n_hid=hdim)

        self.tdnn_init = c_layers.TDNN(idim, hdim)
        assert n_layers > 0
        self.n_layers = n_layers
        self.cells = nn.ModuleDict()
        for i in range(n_layers):
            self.cells[f"tdnn{i}-0"] = c_layers.TDNN(hdim, hdim)
            self.cells[f"tdnn{i}-1"] = c_layers.TDNN(hdim, hdim)
            self.cells[f"lstm{i}"] = c_layers._LSTM(hdim, hdim, 1)
            self.cells[f"bn{i}"] = c_layers.MaskedBatchNorm1d(
                hdim, eps=1e-5, affine=True)
            self.cells[f"dropout{i}"] = nn.Dropout(dropout)

    def impl_forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tmpx, tmp_lens = self.tdnn_init(x, ilens)
        for i in range(self.n_layers):
            tmpx, tmp_lens = self.cells[f"tdnn{i}-0"](tmpx, tmp_lens)
            tmpx, tmp_lens = self.cells[f"tdnn{i}-1"](tmpx, tmp_lens)
            tmpx, tmp_lens = self.cells[f"lstm{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"bn{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"dropout{i}"](tmpx)

        return tmpx, tmp_lens


class BLSTMN(AbsEncoder):
    def __init__(self, idim: int, hdim: int, n_layers: int, dropout: float, num_classes: int = -1, with_head: bool = True) -> None:
        super().__init__(with_head=with_head, num_classes=num_classes, n_hid=hdim)

        assert n_layers > 0
        self.cells = nn.ModuleDict()
        self.n_layers = n_layers
        for i in range(n_layers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim * 2
            self.cells[f"lstm{i}"] = c_layers._LSTM(
                inputdim, hdim, 1, bidirectional=True)
            self.cells[f"bn{i}"] = c_layers.MaskedBatchNorm1d(
                hdim*2, eps=1e-5, affine=True)
            self.cells[f"dropout{i}"] = nn.Dropout(dropout)

    def impl_forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tmp_x, tmp_lens = x, ilens
        for i in range(self.n_layers):
            tmp_x, tmp_lens = self.cells[f"lstm{i}"](tmp_x, tmp_lens)
            tmp_x = self.cells[f"bn{i}"](tmp_x, tmp_lens)
            tmp_x = self.cells[f"dropout{i}"](tmp_x)

        return tmp_x, tmp_lens


class ConformerNet(AbsEncoder):
    """The conformer model with convolution subsampling

    Args:
        num_cells (int): number of conformer blocks
        idim (int): dimension of input features
        hdim (int): hidden size in conformer blocks
        num_classes (int): number of output classes
        conv_multiplier (int): the multiplier to conv subsampling module
        dropout_in (float): the dropout rate to input of conformer blocks (after the linear and subsampling layers)
        res_factor (float): the weighted-factor of residual-connected shortcut in feed-forward module
        d_head (int): dimension of heads in multi-head attention module
        num_heads (int): number of heads in multi-head attention module
        kernel_size (int): kernel size in convolution module
        multiplier (int): multiplier of depth conv in convolution module 
        dropout (float): dropout rate to all conformer internal modules
        delta_feats (bool): True if the input features contains delta and delta-delta features; False if not.
    """

    def __init__(
            self,
            num_cells: int,
            idim: int,
            hdim: int,
            num_classes: int,
            conv: Literal['conv2d', 'vgg2l'] = 'conv2d',
            conv_multiplier: int = None,
            dropout_in: float = 0.2,
            res_factor: float = 0.5,
            d_head: int = 36,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1,
            dropout_attn: float = 0.0,
            delta_feats: bool = False,
            with_head: bool = True,
            subsample_norm: str = 'none'):
        super().__init__(with_head=with_head, num_classes=num_classes, n_hid=hdim)

        if delta_feats:
            in_channel = 3
        else:
            in_channel = 1

        if conv == 'vgg2l':
            self.conv_subsampling = c_layers.VGG2LSubsampling(in_channel)
            ch_sub = math.ceil(math.ceil((idim//in_channel)/2)/2)
            conv_dim = 128 * ch_sub
        elif conv == 'conv2d':
            if conv_multiplier is None:
                conv_multiplier = hdim
            self.conv_subsampling = c_layers.Conv2dSubdampling(
                conv_multiplier, norm=subsample_norm, stacksup=delta_feats)
            conv_dim = conv_multiplier * (((idim//in_channel)//2)//2)
        else:
            raise RuntimeError(f"Unknown type of convolutional layer: {conv}")

        self.linear_drop = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(conv_dim, hdim)),
            ('dropout', nn.Dropout(dropout_in))
        ]))

        self.cells = nn.ModuleList()
        pe = c_layers.PositionalEncoding(hdim)
        for _ in range(num_cells):
            cell = c_layers.ConformerCell(
                hdim, pe, res_factor, d_head, num_heads, kernel_size, multiplier, dropout, dropout_attn)
            self.cells.append(cell)

    def impl_forward(self, x: torch.Tensor, lens: torch.Tensor):
        x_subsampled, ls_subsampled = self.conv_subsampling(x, lens)
        out = self.linear_drop(x_subsampled)
        ls = ls_subsampled
        for cell in self.cells:
            out, ls = cell(out, ls)

        return out, ls


class ConformerLSTM(ConformerNet):
    def __init__(self,
                 hdim_lstm: int,
                 num_lstm_layers: int,
                 dropout_lstm: float,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.lstm = c_layers._LSTM(idim=self.linear_drop.linear.out_channels,
                                   hdim=hdim_lstm, n_layers=num_lstm_layers, dropout=dropout_lstm)

    def impl_forward(self, x: torch.Tensor, lens: torch.Tensor):
        conv_x, conv_ls = super().impl_forward(x, lens)
        return self.lstm(conv_x, conv_ls)

# TODO: (Huahuan) I removed all chunk-related modules.
#       cc @aky15 you may need to add it in v2 standard
