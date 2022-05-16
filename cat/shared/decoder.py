# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Decoder module impl

"""
import kenlm
from typing import *

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

    def __init__(self, num_classes: int = -1, n_emb: int = -1,  n_hid: int = -1, padding_idx: int = -1, tied: bool = False, with_head: bool = True) -> None:
        super().__init__()
        if num_classes == -1:
            return
        if n_hid == -1:
            n_hid = n_emb

        assert num_classes > 0
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

        if not with_head:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(n_hid, num_classes)
            if tied:
                self.classifier.weight = self.embedding.weight

    def score(self, input_ids: torch.LongTensor, targets: torch.LongTensor, input_lengths: Optional[torch.LongTensor] = None, *args):
        if input_lengths is None:
            input_lengths = input_ids.new_ones(
                input_ids.size(0)) * input_ids.size(1)

        # [N, U, K]
        logits, _ = self.forward(input_ids, input_lengths=input_lengths, *args)
        # [N, U]
        log_prob = logits.log_softmax(
            dim=-1).gather(index=targets.unsqueeze(2), dim=-1).squeeze(-1)
        # True for not masked, False for masked, [N, U]
        padding_mask = torch.arange(input_ids.size(1), device=input_ids.device)[
            None, :] < input_lengths[:, None].to(input_ids.device)
        log_prob *= padding_mask
        # [N,]
        score = log_prob.sum(dim=-1)
        return score

    def get_log_prob(self, *args, **kwargs):
        logits, states = self.forward(*args, **kwargs)
        return logits.log_softmax(dim=-1), states

    def batching_states(*args, **kwargs) -> 'AbsStates':
        raise NotImplementedError

    def get_state_from_batch(*args, **kwargs) -> Union['AbsStates', List['AbsStates']]:
        """Get state of given index (or index list) from the batched states"""
        raise NotImplementedError

    def init_states(self, N: int = 1) -> 'AbsStates':
        """The tensor representation of 'None' state of given batch size N"""
        raise NotImplementedError


class AbsStates():
    def __init__(self, state, decoder: AbsDecoder) -> None:
        self._state = state
        self._dec = decoder

    def batching(self, *args, **kwargs):
        return self._dec.batching_states(*args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dec.__class__.__name__})"


class LSTM(AbsDecoder):
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
                 padding_idx: int = -1,
                 with_head: bool = True,
                 *rnn_args, **rnn_kwargs):
        super().__init__(n_emb=hdim, num_classes=num_classes,
                         padding_idx=padding_idx, with_head=with_head)

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
                embedded, input_lengths.to("cpu"), batch_first=True, enforce_sorted=False)
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

    @staticmethod
    def batching_states(states: List[AbsStates]) -> AbsStates:
        h_0 = torch.cat([_state()[0] for _state in states], dim=1)
        c_0 = torch.cat([_state()[1] for _state in states], dim=1)
        return AbsStates((h_0, c_0), LSTM)

    @staticmethod
    def get_state_from_batch(raw_batched_states, index: Union[int, List[int]]) -> Union[AbsStates, List[AbsStates]]:

        if isinstance(index, int):
            flag_squeeze = True
            index = [index]
        else:
            flag_squeeze = False

        o_states = []
        for _i in index:
            h_0 = raw_batched_states[0][:, _i:_i+1, :]
            c_0 = raw_batched_states[1][:, _i:_i+1, :]
            o_states.append(AbsStates((h_0, c_0), LSTM))
        if flag_squeeze:
            return o_states[0]
        else:
            return o_states

    def init_states(self, N: int = 1) -> AbsStates:
        device = next(iter(self.parameters())).device
        h_0 = torch.zeros(
            (self.rnn.num_layers, N, self.rnn.hidden_size), device=device)
        c_0 = torch.zeros_like(h_0)
        return AbsStates((h_0, c_0), self)


class EmbeddingPN(AbsDecoder):
    """Prediction network with embedding layer only."""

    def __init__(self, n_emb: int, num_classes: int = -1, n_hid: int = -1, padding_idx: int = -1, tied: bool = False, with_head: bool = True) -> None:
        super().__init__(n_emb, num_classes=num_classes, n_hid=n_hid,
                         padding_idx=padding_idx, tied=tied, with_head=with_head)
        self.act = nn.ReLU()
        self.with_head = with_head

    def forward(self, x: torch.Tensor, *args, **kwargs):
        embed_x = self.embedding(x)
        if self.with_head:
            return self.classifier(self.act(embed_x)), None
        else:
            return embed_x, None


class CausalTransformer(AbsDecoder):
    def __init__(self,
                 num_classes: int,
                 dim_hid: int,
                 num_head: int,
                 num_layers: int,
                 attn_dropout: float = 0.1,
                 with_head: bool = True,
                 padding_idx: int = -1) -> None:
        super().__init__(n_emb=dim_hid, num_classes=num_classes,
                         padding_idx=padding_idx, with_head=with_head)
        cfg = GPT2Config(
            vocab_size=num_classes, n_embd=dim_hid,
            n_layer=num_layers, n_head=num_head, attn_pdrop=attn_dropout)
        self.trans = GPT2Model(cfg)
        # FIXME (huahun):
        # hacked fix of the issue related to Huggingface,
        # ... see https://github.com/huggingface/transformers/issues/14859
        for name, buffer in self.trans.named_buffers():
            if '.masked_bias' in name:
                buffer.data = torch.tensor(float('-inf'))

        # use my own token embedding layer
        self.trans.wte = None
        self.n_head = num_head
        self.n_layers = num_layers
        self.d_head = dim_hid//num_head

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

    @staticmethod
    def batching_states(states: List[AbsStates]) -> AbsStates:
        if states[0]() is None:
            for _state in states:
                assert _state() is None
            return AbsStates(None, CausalTransformer)

        n_layers = len(states[0]())
        batched_states = []
        for l in range(n_layers):
            _state_0 = torch.cat([_state()[l][0] for _state in states], dim=0)
            _state_1 = torch.cat([_state()[l][1] for _state in states], dim=0)
            batched_states.append((_state_0, _state_1))

        return AbsStates(tuple(batched_states), CausalTransformer)

    @staticmethod
    def get_state_from_batch(raw_batched_states, index: Union[int, List[int]]) -> Union[AbsStates, List[AbsStates]]:

        if isinstance(index, int):
            flag_squeeze = True
            index = [index]
        else:
            flag_squeeze = False

        n_layers = len(raw_batched_states)
        o_states = []
        for _i in index:
            _o_state = []
            for l in range(n_layers):
                s_0 = raw_batched_states[l][0][_i:_i+1, :, :, :]
                s_1 = raw_batched_states[l][1][_i:_i+1, :, :, :]
                _o_state.append((s_0, s_1))
            o_states.append(AbsStates(tuple(_o_state), CausalTransformer))

        if flag_squeeze:
            return o_states[0]
        else:
            return o_states

    def init_states(self, N: int = 1) -> AbsStates:
        return AbsStates(None, CausalTransformer)


class NGram(AbsDecoder):
    def __init__(self,
                 gram_order: int,
                 num_classes: int,
                 f_binlm: str,
                 bos_id: int = 0,
                 eos_id: int = -1,
                 unk_id: int = 1) -> None:
        super().__init__()
        self.gram_order = gram_order
        self.vocab = {x: str(x) for x in range(num_classes)}
        # set 0 -> </s>, 1 -> <unk>
        if eos_id == -1:
            eos_id = bos_id
        self.vocab[bos_id] = '<s>'
        self.vocab[eos_id] = '</s>'
        self.vocab[unk_id] = '<unk>'
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.ngram = kenlm.Model(f_binlm)
        # scale: convert log10 -> loge
        self.register_buffer('scale', torch.tensor(
            10.).log_(), persistent=False)

    def score(self, input_ids: torch.LongTensor, targets: torch.LongTensor, input_lengths: Optional[torch.LongTensor] = None):
        input_ids = input_ids.cpu()
        targets = targets.cpu()
        if input_lengths is None:
            in_lens = [input_ids.size(1)]*input_ids.size(0)
        else:
            in_lens = input_lengths.cpu().tolist()

        # [N, ]
        log_prob = input_ids.new_full(
            input_ids.size()[:1], 0.0, dtype=torch.float)
        for b in range(input_ids.size(0)):
            """
            NOTE (huahuan): For n-gram model, we assume the input_ids[:, 1:] == targets[:, :-1]
            """
            seq_str = [self.vocab[i]
                       for i in input_ids[b, :in_lens[b]].tolist()]
            # replace </s> in the first place to <s>
            if seq_str[0] == '</s>':
                seq_str[0] = '<s>'
            # add last token, usually </s>
            seq_str.append(self.vocab[targets[b][-1].item()])
            log_prob[b] = self.ngram.score(
                ' '.join(seq_str), bos=False, eos=False)

        log_prob *= self.scale

        return log_prob

    def forward(self, src_ids: torch.Tensor, hidden: torch.Tensor = None, input_lengths: Optional[torch.Tensor] = None):
        """This is non-standar interface, only designed for inference. The n-gram model will take input as context and 
            predict the probability for next token, so the output is always (N, 1, V)
        """
        if self.training:
            raise NotImplementedError(
                "N-gram model doesn't support training like NN model.")

        if input_lengths is not None and src_ids.size(0) > 1:
            raise NotImplementedError(
                "N-gram model for batched sequences likelihood calculation is of poor efficiency.")

        B = src_ids.size(0)
        if hidden is not None:
            assert hidden.size(0) == B
            input_ids = torch.cat([hidden, src_ids], dim=1)
        else:
            input_ids = src_ids

        # keep N-1 ids
        input_ids = input_ids[:, -(self.gram_order-1):]

        pred_logp = [[] for _ in range(B)]
        for b, seq in enumerate(input_ids.cpu().tolist()):
            seq = [self.vocab[x] for x in seq]
            state = init_state(self.ngram, seq)
            for tok in self.vocab.values():
                pred_logp[b].append(update_state(self.ngram, state, tok)[0])

        return self.scale*self.scale.new_tensor(pred_logp).unsqueeze(1), input_ids

    @staticmethod
    def batching_states(states: List[AbsStates]) -> AbsStates:
        if states[0]() is None:
            for _state in states:
                assert _state() is None
            return AbsStates(None, NGram)
        o_state = torch.cat([_s() for _s in states], dim=0)
        return AbsStates(o_state, NGram)

    @staticmethod
    def get_state_from_batch(raw_batched_states, index: Union[int, List[int]]) -> Union[AbsStates, List[AbsStates]]:

        if isinstance(index, int):
            return AbsStates(raw_batched_states[index:index+1, :], NGram)
        else:
            return [AbsStates(raw_batched_states[i:i+1, :], NGram) for i in index]

    def init_states(self, N: int = 1):
        return AbsStates(None, self)


class ZeroDecoder(AbsDecoder):
    def __init__(self, hdim: int, *args, **kwargs) -> None:
        super().__init__()
        self._dummy_hdim = hdim

    def forward(self, x: torch.Tensor, *args):
        return torch.zeros_like(x).unsqueeze_(2).repeat(1, 1, self._dummy_hdim), None

    def score(self, *args):
        raise NotImplementedError

    @staticmethod
    def batching_states(*args, **kwargs) -> 'AbsStates':
        return AbsStates(None, ZeroDecoder)

    @staticmethod
    def get_state_from_batch(raw_batched_states, index: Union[int, List[int]]) -> Union[AbsStates, List[AbsStates]]:

        if isinstance(index, int):
            return AbsStates(None, ZeroDecoder)
        else:
            return [AbsStates(None, ZeroDecoder) for i in index]

    def init_states(self, N: int = 1) -> 'AbsStates':
        return AbsStates(None, ZeroDecoder)


class ILM(AbsDecoder):
    """
    ILM estimation of RNN-T, referring to
    "Internal Language Model Estimation for Domain-Adaptive End-to-End Speech Recognition"
    https://arxiv.org/abs/2011.01991
    """

    def __init__(self, f_rnnt_config: str, f_check: str):
        super().__init__()
        from cat.rnnt import rnnt_builder
        from cat.shared import coreutils
        cfg = coreutils.readjson(f_rnnt_config)
        rnntmodel = rnnt_builder(cfg, dist=False)
        coreutils.load_checkpoint(rnntmodel, f_check)
        self._stem = rnntmodel.decoder
        self._head = rnntmodel.joiner
        self._dim_enc_out = cfg['joiner']['kwargs']['odim_enc']
        del rnntmodel

    def forward(self, x, input_lengths):
        # [N, U, H]
        decoder_out, _ = self._stem(x, input_lengths=input_lengths)
        # [N, U, H] + [N, 1, H] -> [N, 1, U, V] -> [N, U, V]
        logits = self._head.impl_forward(
            decoder_out.new_zeros((decoder_out.size(0), 1, self._dim_enc_out)),
            decoder_out).squeeze(1)
        logits[:, :, 0].fill_(logits.min() - 1e9)
        return logits, None


class MultiDecoder(AbsDecoder):
    def __init__(self, weights: List[float], f_configs: List[str], f_checks: Optional[List[Union[str, None]]] = None) -> None:
        super().__init__()

        assert len(weights) == len(f_configs)
        if f_checks is None:
            f_checks = [None]*len(weights)
        else:
            assert len(weights) == len(f_checks)
        self._num_decs = len(weights)

        from cat.lm import lm_builder
        from cat.shared import coreutils

        self._weights = weights
        self._decs = nn.ModuleList()
        zeroweight = []
        for i in range(self._num_decs):
            if self._weights[i] == 0.:
                zeroweight.append(i)
                continue
            _dec = lm_builder(coreutils.readjson(
                f_configs[i]), dist=False, wrapper=True)
            if f_checks[i] is not None:
                coreutils.load_checkpoint(_dec, f_checks[i])
            self._decs.append(_dec.lm)
        for i in zeroweight[::-1]:
            self._weights.pop(i)
        self._num_decs -= len(zeroweight)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_log_prob(self, x, hidden, *args, **kwargs):
        out = 0.
        state = []
        for i in range(self._num_decs):
            part_out, part_state = self._decs[i].get_log_prob(
                x, hidden[i], *args, **kwargs)
            out += part_out * self._weights[i]
            state.append(part_state)
        return out, tuple(state)

    def batching_states(self, states: List[AbsStates]) -> AbsStates:
        shard_states = list(zip(*[s() for s in states]))
        assert len(shard_states) == self._num_decs

        batched_state = tuple(
            self._decs[i].batching_states(
                [
                    AbsStates(shard_shard_state, self._decs[i])
                    for shard_shard_state in shard_states[i]
                ]
            )()
            for i in range(self._num_decs)
        )
        return AbsStates(batched_state, self)

    def get_state_from_batch(self, raw_batched_states: Tuple[Any], index: Union[int, List[int]]) -> Union[AbsStates, List[AbsStates]]:
        if isinstance(index, int):
            return AbsStates(
                tuple(
                    self._decs[i].get_state_from_batch(
                        raw_batched_states[i], index)()
                    for i in range(self._num_decs)
                ), self)
        else:
            return [
                AbsStates(
                    tuple(
                        self._decs[i].get_state_from_batch(
                            raw_batched_states[i], _idx)()
                        for i in range(self._num_decs)
                    ), self)
                for _idx in index
            ]

    def init_states(self, N: int = 1) -> 'AbsStates':
        return AbsStates(
            tuple(
                self._decs[i].init_states(N)()
                for i in range(self._num_decs)
            ), self)


def init_state(model: kenlm.Model, pre_toks: List[str]):
    state, state2 = kenlm.State(),  kenlm.State()
    for tok in pre_toks:
        model.BaseScore(state, tok, state2)
        state, state2 = state2, state
    return state


def update_state(model: kenlm.Model, prev_state: kenlm.State, token: str):
    new_state = kenlm.State()
    log_p = model.BaseScore(prev_state, token, new_state)
    return log_p, new_state
