"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Keyu An, Huahuan Zheng

In this file, we define universal models 
"""

import numpy as np
import _layers
from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn


def get_vgg2l_odim(idim, in_channel=1, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


class LSTM(nn.Module):
    def __init__(self,
                 idim: int,
                 hdim: int,
                 n_layers: int,
                 num_classes: int,
                 dropout: float,
                 bidirectional: bool = False):
        super().__init__()
        self.lstm = _layers._LSTM(
            idim, hdim, n_layers, dropout, bidirectional=bidirectional)

        if bidirectional:
            self.linear = nn.Linear(hdim * 2, num_classes)
        else:
            self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        lstm_out, olens = self.lstm(x, ilens, hidden)
        out = self.linear(lstm_out)
        return out, olens


class BLSTM(LSTM):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int, dropout: float):
        super().__init__(idim, hdim, n_layers, num_classes, dropout, bidirectional=True)


class VGGLSTM(LSTM):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int, dropout: float, in_channel: int = 3, bidirectional: int = False):
        super().__init__(get_vgg2l_odim(idim, in_channel=in_channel), hdim,
                         n_layers, num_classes, dropout, bidirectional=bidirectional)

        self.VGG = _layers.VGG2L(in_channel)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        vgg_o, vgg_lens = self.VGG(x, ilens)
        return super().forward(vgg_o, vgg_lens)


class VGGBLSTM(VGGLSTM):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int, dropout: float, in_channel: int = 3):
        super().__init__(idim, hdim, n_layers, num_classes,
                         dropout, in_channel=in_channel, bidirectional=True)


class LSTMrowCONV(nn.Module):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int, dropout: float):
        super().__init__()

        self.lstm = _layers._LSTM(idim, hdim, n_layers, dropout)
        self.lookahead = _layers.Lookahead(hdim, context=5)
        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        lstm_out, olens = self.lstm(x, ilens, hidden)
        ahead_out = self.lookahead(lstm_out)
        return self.linear(ahead_out), olens


class TDNN_NAS(torch.nn.Module):
    def __init__(self, idim: int, hdim: int,  num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.tdnns = nn.ModuleDict(OrderedDict([
            ('tdnn0', _layers.TDNN(idim, hdim, half_context=2, dilation=1)),
            ('tdnn1', _layers.TDNN(idim, hdim, half_context=2, dilation=2)),
            ('tdnn2', _layers.TDNN(idim, hdim, half_context=2, dilation=1)),
            ('tdnn3', _layers.TDNN(idim, hdim, stride=3)),
            ('tdnn4', _layers.TDNN(idim, hdim, half_context=2, dilation=2)),
            ('tdnn5', _layers.TDNN(idim, hdim, half_context=2, dilation=1)),
            ('tdnn6', _layers.TDNN(idim, hdim, half_context=2, dilation=2))
        ]))

        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tmp_x, tmp_lens = x, ilens
        for i, tdnn in enumerate(self.tdnns.values):
            if i < len(self.tdnns)-1:
                tmp_x = self.dropout(x)
            tmp_x, tmp_lens = tdnn(tmp_x, tmp_lens)

        return self.linear(tmp_x), tmp_lens


class TDNN_LSTM(torch.nn.Module):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int,  dropout: float):
        super().__init__()

        self.tdnn_init = _layers.TDNN(idim, hdim)
        assert n_layers > 0
        self.n_layers = n_layers
        self.cells = nn.ModuleDict()
        for i in range(n_layers):
            self.cells[f"tdnn{i}-0"] = _layers.TDNN(hdim, hdim)
            self.cells[f"tdnn{i}-1"] = _layers.TDNN(hdim, hdim)
            self.cells[f"lstm{i}"] = _layers._LSTM(hdim, hdim, 1)
            self.cells[f"bn{i}"] = _layers.MaskedBatchNorm1d(
                hdim, eps=1e-5, affine=True)
            self.cells[f"dropout{i}"] = nn.Dropout(dropout)

        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):

        tmp_x, tmp_lens = self.tdnn_init(x, ilens)

        for i in range(self.n_layers):
            tmpx, tmp_lens = self.cells[f"tdnn{i}-0"](tmpx, tmp_lens)
            tmpx, tmp_lens = self.cells[f"tdnn{i}-1"](tmpx, tmp_lens)
            tmpx, tmp_lens = self.cells[f"lstm{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"bn{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"dropout{i}"](tmpx)

        return self.linear(tmpx), tmp_lens


class BLSTMN(torch.nn.Module):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int,  dropout: float):
        super(BLSTMN, self).__init__()
        assert n_layers > 0
        self.cells = nn.ModuleDict()
        self.n_layers = n_layers
        for i in range(n_layers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim * 2
            self.cells[f"lstm{i}"] = _layers._LSTM(
                inputdim, hdim, 1, bidirectional=True)
            self.cells[f"bn{i}"] = _layers.MaskedBatchNorm1d(
                hdim*2, eps=1e-5, affine=True)
            self.cells[f"dropout{i}"] = nn.Dropout(dropout)

        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tmp_x, tmp_lens = x, ilens
        for i in range(self.n_layers):
            tmp_x, tmp_lens = self.cells[f"lstm{i}"](tmp_x, tmp_lens)
            tmp_x = self.cells[f"bn{i}"](tmp_x, tmp_lens)
            tmp_x = self.cells[f"dropout{i}"](tmp_x)

        return self.linear(tmp_x), tmp_lens


class ConformerNet(nn.Module):
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
            subsample_norm: str = 'none'):
        super().__init__()

        if delta_feats:
            in_channel = 3
        else:
            in_channel = 1

        if conv == 'vgg2l':
            self.conv_subsampling = _layers.VGG2LSubsampling(in_channel)
            conv_dim = 128 * (idim//in_channel//4)
        elif conv == 'conv2d':
            if conv_multiplier is None:
                conv_multiplier = hdim
            self.conv_subsampling = _layers.Conv2dSubdampling(
                conv_multiplier, norm=subsample_norm, stacksup=delta_feats)
            conv_dim = conv_multiplier * (idim//in_channel//4)
        else:
            raise RuntimeError(f"Unknown type of convolutional layer: {conv}")

        self.linear_drop = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(conv_dim, hdim)),
            ('dropout', nn.Dropout(dropout_in))
        ]))

        self.cells = nn.ModuleList()
        pe = _layers.PositionalEncoding(hdim)
        for _ in range(num_cells):
            cell = _layers.ConformerCell(
                hdim, pe, res_factor, d_head, num_heads, kernel_size, multiplier, dropout, dropout_attn)
            self.cells.append(cell)

        self.classifier = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        x_subsampled, ls_subsampled = self.conv_subsampling(x, lens)
        out = self.linear_drop(x_subsampled)
        ls = ls_subsampled
        for cell in self.cells:
            out, ls = cell(out, ls)
        logits = self.classifier(out)

        return logits, ls


class ConformerLSTM(ConformerNet):
    def __init__(self,
                 hdim_lstm: int,
                 num_lstm_layers: int,
                 dropout_lstm: float,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.lstm = _layers._LSTM(idim=self.linear_drop.linear.out_channels,
                                   hdim=hdim_lstm, n_layers=num_lstm_layers, dropout=dropout_lstm)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        conv_x, conv_ls = super().forward(x, lens)
        return self.lstm(conv_x, conv_ls)

# TODO: (Huahuan) I removed all chunk-related modules.
#       cc @aky15 you may need to add it in v2 standard
