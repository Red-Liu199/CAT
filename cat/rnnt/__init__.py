# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""RNN-Transducer related module
"""


from ..shared import coreutils as utils

import gather
from typing import Union, Tuple, Sequence, Literal, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


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
                self._data = gather.cat(xs, xn)
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
        with autocast(enabled=False):
            return gather.sum(self._data.float(), _y._data.float(), self._lens, _y._lens)


class SimJointNet(nn.Module):
    def __init__(self, dim_tn: int, dim_pn: int, num_classes: int):
        super().__init__()
        assert dim_tn > 0 and isinstance(dim_tn, int)
        assert dim_pn > 0 and isinstance(dim_pn, int)
        assert num_classes > 0 and isinstance(num_classes, int)

        self._skip_softmax = False
        if dim_tn == num_classes:
            self.fc_tn = None
        else:
            self.fc_tn = nn.Linear(dim_tn, num_classes)

        if dim_pn == num_classes:
            self.fc_pn = None
        else:
            self.fc_pn = nn.Linear(dim_pn, num_classes)

    def forward(self, tn_out: Union[torch.Tensor, PackedSequence], pn_out: Union[torch.Tensor, PackedSequence]) -> torch.FloatTensor:
        assert not (isinstance(tn_out, PackedSequence) ^
                    isinstance(pn_out, PackedSequence))

        if isinstance(tn_out, PackedSequence):
            if self.fc_pn is not None:
                pn_out = pn_out.set(self.fc_pn(pn_out.data))

            if self.fc_tn is not None:
                tn_out = tn_out.set(self.fc_tn(tn_out.data))

        else:
            if self.fc_pn is not None:
                pn_out = self.fc_pn(pn_out)

            if self.fc_tn is not None:
                tn_out = self.fc_tn(tn_out)

            if pn_out.dim() == 3:
                pn_out = pn_out.unsqueeze(1)
                tn_out = tn_out.unsqueeze(2)

        if self._skip_softmax:
            return pn_out + tn_out
        else:
            expand_out = (pn_out+tn_out).log_softmax(dim=-1)
            return expand_out

    def skip_softmax_forward(self, *args, **kwargs):
        self._skip_softmax = True
        outs = self.forward(*args, **kwargs)
        self._skip_softmax = False
        return outs


class JointNet(nn.Module):
    """
    Joint `encoder_output` and `decoder_output`.
    Args:
        encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size ``(batch, time_steps, dimensionA)``
        decoder_output (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size ``(batch, label_length, dimensionB)``
    Returns:
        outputs (torch.FloatTensor): outputs of joint `encoder_output` and `decoder_output`. `FloatTensor` of size ``(batch, time_steps, label_length, dimensionA + dimensionB)``
    """

    def __init__(self,
                 odim_encoder: int,
                 odim_decoder: int,
                 num_classes: int,
                 hdim: int = -1,
                 HAT: bool = False,
                 joint_mode: Literal['add', 'cat'] = 'add',
                 act: Literal['tanh', 'relu'] = 'tanh'):
        super().__init__()
        self._skip_softmax = False

        self._isHAT = HAT
        if act == 'tanh':
            act_layer = nn.Tanh()
        elif act == 'relu':
            act_layer = nn.ReLU()
        else:
            raise NotImplementedError(f"Unknown activation layer type: {act}")

        if joint_mode == 'add':
            if hdim == -1:
                hdim = max(odim_decoder, odim_encoder)
            self.fc_enc = nn.Linear(odim_encoder, hdim)
            self.fc_dec = nn.Linear(odim_decoder, hdim)
            self.fc = nn.Sequential(
                act_layer,
                nn.Linear(hdim, num_classes)
            )
        elif joint_mode == 'cat':
            self.fc_enc = None
            self.fc_dec = None
            self.fc = nn.Sequential(
                act_layer,
                nn.Linear(odim_encoder + odim_decoder, num_classes)
            )
        else:
            raise RuntimeError(f"Unknown mode for joint net: {joint_mode}")

        self._mode = joint_mode
        self._V = num_classes

        if HAT:
            """
            Implementation of Hybrid Autoregressive Transducer (HAT)
            https://arxiv.org/abs/2003.07705
            """
            self.distr_blk = nn.Sigmoid()

    def forward(self, encoder_output: Union[torch.Tensor, PackedSequence], decoder_output: Union[torch.Tensor, PackedSequence]) -> torch.FloatTensor:

        if isinstance(encoder_output, PackedSequence) and isinstance(decoder_output, PackedSequence):
            if self._mode == 'add':
                # compact memory mode, gather the tensors without padding
                encoder_output.set(self.fc_enc(encoder_output.data))
                decoder_output.set(self.fc_dec(decoder_output.data))
            else:
                dim_enc, dim_dec = encoder_output.data.size(
                    -1), decoder_output.data.size(-1)
                encoder_output.set(torch.nn.functional.pad(
                    encoder_output.data, (0, dim_dec)))
                decoder_output.set(torch.nn.functional.pad(
                    decoder_output.data, (dim_enc, 0)))

            # shape: (\sum_{Ti(Ui+1)}, V)
            expanded_out = encoder_output + decoder_output

        elif isinstance(encoder_output, torch.Tensor) and isinstance(decoder_output, torch.Tensor):

            assert (encoder_output.dim() == 3 and decoder_output.dim() == 3) or (
                encoder_output.dim() == 1 and decoder_output.dim() == 1)

            if self._mode == 'add':
                encoder_output = self.fc_enc(encoder_output)
                decoder_output = self.fc_dec(decoder_output)

            if encoder_output.dim() == 3:
                _, T, _ = encoder_output.size()
                _, Up, _ = decoder_output.size()
                encoder_output = encoder_output.unsqueeze(2)
                decoder_output = decoder_output.unsqueeze(1)
                encoder_output = encoder_output.expand(-1, -1, Up, -1)
                decoder_output = decoder_output.expand(-1, T, -1, -1)

            if self._mode == 'add':
                expanded_out = encoder_output + decoder_output
            else:
                expanded_out = torch.cat(
                    [encoder_output, decoder_output], dim=-1)

        else:
            raise NotImplementedError(
                "Output of encoder and decoder being fed into jointnet should be of same type. Expect (Tensor, Tensor) or (PackedSequence, PackedSequence), instead ({}, {})".format(type(encoder_output), type(decoder_output)))

        if self._skip_softmax and self._isHAT:
            raise RuntimeError("HAT mode is not supprot with skip softmax.")
        if self._isHAT:
            vocab_logits = self.fc(expanded_out)
            prob_blk = self.distr_blk(vocab_logits[..., :1])
            vocab_log_probs = torch.log(
                1-prob_blk)+torch.log_softmax(vocab_logits[..., 1:], dim=-1)
            outputs = torch.cat([torch.log(prob_blk), vocab_log_probs], dim=-1)
        else:
            if self._skip_softmax:
                outputs = self.fc(expanded_out)
            else:
                outputs = self.fc(expanded_out).log_softmax(dim=-1)

        return outputs

    def skip_softmax_forward(self, *args, **kwargs):
        self._skip_softmax = True
        outs = self.forward(*args, **kwargs)
        self._skip_softmax = False
        return outs


from .train import build_model as rnnt_builder