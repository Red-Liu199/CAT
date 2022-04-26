"""Beam search for Transducer sequence.

Support:

- native decoding (with prefix tree merge)
- latency controlled decoding (with prefix tree merge)
- alignment-length synchronous decoding

Author: Huahuan Zhengh (maxwellzh@outlook.com)
"""

from .joiner import AbsJointNet
from ..shared import coreutils
from ..shared.decoder import (
    LSTM,
    AbsDecoder,
    AbsStates
)

import os
from typing import *

import torch


def logaddexp(a: torch.Tensor, b: torch.Tensor):
    if a.dtype == torch.float:
        return torch.logaddexp(a, b)
    elif a.dtype == torch.half:
        if a < b:
            a, b = b, a
        # a + log(1 + exp(b-a))
        return a + (1 + (b-a).exp()).log()
    else:
        raise ValueError


def fclone(fp: Union[float, torch.FloatTensor]):
    if isinstance(fp, float):
        return fp
    elif isinstance(fp, torch.Tensor):
        return fp.clone()
    else:
        raise NotImplementedError(f"Invalid type of float point {fp}")


def hash_tensor(t: torch.LongTensor) -> Tuple[int]:
    return tuple(t.cpu().tolist())


class Hypothesis():
    def __init__(
            self,
            pred: torch.LongTensor,
            log_prob: Union[torch.Tensor, float],
            cache: Union[Dict[str, Union[AbsStates, torch.Tensor]], None],
            lm_score: Union[torch.Tensor, float] = 0.0,
            len_norm: bool = False) -> None:

        self._last_token = pred[-1:]
        self.pred = hash_tensor(pred)
        self.log_prob = log_prob
        self.cache = cache
        self.lm_score = lm_score
        self.len_norm = len_norm
        self.ilm_log_prob = 0.0

    def pop_last(self) -> torch.LongTensor:
        return self._last_token

    @property
    def score(self):
        score = self.log_prob + self.lm_score
        if self.len_norm:
            return score/len(self)
        else:
            return score

    def get_pred_token(self, return_tensor: bool = False):
        if return_tensor:
            return self._last_token.new_tensor(self.pred)
        else:
            return list(self.pred)

    def clone(self):
        new_hypo = Hypothesis(
            self._last_token.clone(),
            fclone(self.log_prob),
            self.cache.copy(),
            fclone(self.lm_score),
            self.len_norm
        )
        new_hypo.pred = self.pred[:]
        new_hypo.ilm_log_prob = fclone(self.ilm_log_prob)
        return new_hypo

    def __add__(self, rhypo):
        new_hypo = self.clone()
        new_hypo.log_prob = logaddexp(new_hypo.log_prob, rhypo.log_prob)
        return new_hypo

    def add_(self, rhypo):
        '''in-place version of __add__'''
        self.log_prob = logaddexp(self.log_prob, rhypo.log_prob)
        return self

    def add_token(self, tok: torch.LongTensor):
        self._last_token = tok.view(1)
        self.pred += (tok.item(),)

    def __len__(self) -> int:
        return len(self.pred)

    def __repr__(self) -> str:
        return f"Hypothesis({self.pred}, score={self.score:.2f})"


class PrefixCacheDict():
    """
    This use a map-style way to store the cache.
    Compared to tree-like structure, thie would be less efficient when the tree is 
    quite large. But more efficient when it is small.
    """

    def __init__(self) -> None:
        self._cache = {}    # type: Dict[Tuple[int], Dict]

    def __contains__(self, pref: Tuple[int]) -> bool:
        return pref in self._cache

    def update(self, pref: Tuple[int], new_cache: dict):
        if pref in self._cache:
            self._cache[pref].update(new_cache.copy())
        else:
            self._cache[pref] = new_cache.copy()

    def getCache(self, pref: Tuple[int]) -> Union[None, dict]:
        '''Get cache. If there isn't such prefix, return None.
        '''
        if pref in self._cache:
            return self._cache[pref]
        else:
            return None

    def pruneAllBut(self, legal_prefs: List[Tuple[int]]):
        new_cache = {}
        for pref in legal_prefs:
            if pref in self._cache:
                new_cache[pref] = self._cache[pref]
        del self._cache
        self._cache = new_cache

    def pruneShorterThan(self, L: int):
        torm = [key for key in self._cache if len(key) < L]
        for k in torm:
            del self._cache[k]

    def __str__(self) -> str:
        cache = {}
        for k in self._cache.keys():
            cache[k] = {}
            for _k in self._cache[k].keys():
                cache[k][_k] = '...'

        return str(cache)


def beam_append(ongoing_beams: List[Hypothesis], new_beam: Hypothesis, prefix_merge: bool = False) -> List[Hypothesis]:
    """Append the new hypothesis into ongoing_beams w/ or w/o prefix merging"""
    if prefix_merge:
        for _beam in ongoing_beams:
            if _beam.pred == new_beam.pred:
                _beam.add_(new_beam)
                return ongoing_beams

    ongoing_beams.append(new_beam)
    return ongoing_beams


def recombine_hypo(redundant_hypos: List[Hypothesis]) -> List[Hypothesis]:
    """Recombine the hypos to merge duplicate path"""
    out_hypos = {}  # type: Dict[str, Hypothesis]
    for hypo in redundant_hypos:
        if hypo.pred in out_hypos:
            out_hypos[hypo.pred].add_(hypo)
        else:
            out_hypos[hypo.pred] = hypo

    return list(out_hypos.values())

# TODO:
# 1. add a interface of decoder
# 2. batch-fly the decoding
# 3. interface for introducing external LM(s)
# 3. rename tn -> encoder


class BeamSearcher():

    def __init__(
        self,
        predictor: AbsDecoder,
        joiner: AbsJointNet,
        blank_id: int = 0,
        bos_id: int = 0,
        beam_size: int = 5,
        nbest: int = -1,
        lm_module: Optional[AbsDecoder] = None,
        alpha: Optional[float] = 0.,
        beta: Optional[float] = 0.,
        est_ilm: bool = False
    ):
        super(BeamSearcher, self).__init__()
        assert blank_id == bos_id

        if alpha == 0.0:
            # NOTE: alpha = 0 will disable LM interation whatever beta is.
            beta = None
            alpha = None
            lm_module = None

        if lm_module is None:
            alpha = 0.0
            beta = 0.0

        self.predictor = predictor
        self.joiner = joiner
        self.blank_id = blank_id
        self.bos_id = bos_id
        self.beam_size = beam_size
        if nbest == -1:
            nbest = beam_size
        self.nbest = min(nbest, beam_size)
        self.lm = lm_module
        self.alpha_ = alpha

        self.beta_ = beta
        self.est_ilm = est_ilm

        self.searcher = self.batching_rna

    def __call__(self, enc_out: torch.Tensor, frame_lens: Optional[torch.Tensor] = None) -> List[Tuple[List[List[int]], List[float], Union[None, List[float]]]]:
        hypos = self.searcher(enc_out, frame_lens)

        return [
            (
                [hypo.get_pred_token()[1:] for hypo in _hyps],
                [hypo.score.item() for hypo in _hyps],
                [hypo.ilm_log_prob.item()
                 for hypo in _hyps] if self.est_ilm else None
            ) for _hyps in hypos]

    def batching_rna(self, encoder_out: torch.Tensor, frame_lengths: Optional[torch.Tensor] = None) -> List[List[Hypothesis]]:
        """
        An implementation of batched RNA decoding

        encoder_out: (N, T, V)
        """
        use_lm = self.lm is not None
        if isinstance(self.predictor, LSTM):
            if use_lm and not isinstance(self.lm, LSTM):
                fixlen_state = False
            else:
                fixlen_state = True
        else:
            fixlen_state = False

        n_batches, n_max_frame_length = encoder_out.shape[:2]
        dummy_token = encoder_out.new_empty(1, dtype=torch.long)
        idx_seq = torch.arange(n_batches)
        if frame_lengths is None:
            frame_lengths = dummy_token.new_full(
                n_batches, fill_value=n_max_frame_length)
        else:
            frame_lengths = frame_lengths.clone()

        Beams = [[Hypothesis(
            pred=dummy_token.new_tensor([self.bos_id]),
            log_prob=0.0,
            cache={'pn_state': self.predictor.init_states()})]
            for _ in range(n_batches)]

        if use_lm:
            for b in range(n_batches):
                Beams[b][0].cache.update({'lm_state': self.lm.init_states()})
        prefix_cache = PrefixCacheDict()

        for t in range(n_max_frame_length):
            # concat beams in the batch to one group
            idx_ongoing_seq = idx_seq[frame_lengths > 0]
            n_seqs = idx_ongoing_seq.size(0)
            batched_beams = sum((Beams[i_]
                                for i_ in idx_ongoing_seq), [])
            # n_beams = len(batched_beams)
            group_uncached, group_cached = group_to_batch(
                batched_beams,
                dummy_token,
                prefix_cache,
                statelen_fixed=fixlen_state)
            group_beams = group_uncached + group_cached

            idxbeam2srcidx = []   # len: n_beams
            group_pn_out = []     # len: len(group_beams)
            group_lm_out = []

            n_group_uncached = len(group_uncached)
            # In following loop, we do:
            # 1. compute predictor output for beams not in cache
            # 2. fetch output in cache for beams in cache
            for i, (g_index, g_tokens, g_states) in enumerate(group_beams):
                idxbeam2srcidx += g_index
                if i < n_group_uncached:
                    pn_out, pn_state = self.predictor(
                        g_tokens, g_states['pn_state']())
                    if use_lm:
                        lm_out, lm_state = self._lm_step(
                            g_tokens, g_states['lm_state']())
                    # add into cache
                    for bid, absidx in enumerate(g_index):
                        cur_cache = {
                            'pn_out': pn_out[bid:bid+1],
                            'pn_state': self.predictor.get_state_from_batch(pn_state, bid)
                        }
                        if use_lm:
                            cur_cache.update({
                                'lm_out': lm_out[bid:bid+1],
                                'lm_state': self.lm.get_state_from_batch(lm_state, bid)
                            })
                        prefix_cache.update(
                            batched_beams[absidx].pred, cur_cache)
                else:
                    pn_out = g_tokens['pn_out']
                    pn_state = g_states['pn_state']()
                    if use_lm:
                        lm_out = g_tokens['lm_out']
                        lm_state = g_states['lm_state']()

                group_pn_out.append(pn_out)
                if use_lm:
                    group_lm_out.append(lm_out)

            # pn_out: (n_beams, 1, H)
            pn_out = torch.cat(group_pn_out, dim=0)
            # expand_tn_out: (n_beams, 1, H)
            expand_enc_out = torch.cat(
                [encoder_out[b:b+1, t:t+1, :].expand(len(Beams[b]), -1, -1)
                 for b in idx_ongoing_seq], dim=0)[idxbeam2srcidx]
            # log_prob: (n_beams, 1, 1, V) -> (n_beams, V)
            log_prob = self.joiner(
                expand_enc_out, pn_out).squeeze(1).squeeze(1)

            if self.est_ilm:
                ilm_log_prob = self.joiner.impl_forward(
                    torch.zeros_like(expand_enc_out), pn_out).squeeze(1).squeeze(1)
                if self.blank_id == 0:
                    ilm_log_prob[:, 1:] = \
                        ilm_log_prob[:, 1:].log_softmax(dim=1)
                else:
                    raise NotImplementedError

            if use_lm:
                # lm_out: (n_beams, 1, V)
                lm_out = torch.cat(group_lm_out, dim=0)
                lm_score = self.alpha_ * \
                    lm_out.squeeze(1) + self.beta_
                # combine_score: (n_beams, V)
                combine_score = log_prob + lm_score
                combine_score[:, self.blank_id] = log_prob[:, self.blank_id]
            else:
                combine_score = log_prob.clone()

            V = combine_score.size(-1)
            combine_score += collect_scores(
                [batched_beams[b] for b in idxbeam2srcidx], dummy_token).unsqueeze(1)
            offset = 0
            min_len = n_max_frame_length
            srcidx2beamidx = {i_: idx for idx, i_ in enumerate(idxbeam2srcidx)}
            for s_ in range(n_seqs):
                idxinbatch = idx_ongoing_seq[s_]
                map2rearangeidx = [srcidx2beamidx[offset+beamidx]
                                   for beamidx in range(len(Beams[idxinbatch]))]
                # flattened_pos: (K, )
                _, flattened_pos = torch.topk(
                    combine_score[map2rearangeidx].flatten(),
                    k=(self.beam_size))
                # idx_beam, tokens: (K, )
                idx_beam = [map2rearangeidx[i_] for i_ in torch.div(
                    flattened_pos, V, rounding_mode='floor')]
                tokens = flattened_pos % V
                A = []
                for tok_, b in zip(tokens, idx_beam):
                    cur_hypo = batched_beams[idxbeam2srcidx[b]].clone()
                    cur_hypo.log_prob += log_prob[b, tok_]
                    if self.est_ilm:
                        cur_hypo.ilm_log_prob += ilm_log_prob[b, tok_]
                    if tok_ == self.blank_id:
                        A.append(cur_hypo)
                        continue
                    # the order of following two lines cannot be changed
                    cur_hypo.cache = prefix_cache.getCache(cur_hypo.pred)
                    cur_hypo.add_token(tok_)
                    if use_lm:
                        cur_hypo.lm_score += lm_score[b, tok_]
                    A.append(cur_hypo)

                A = recombine_hypo(A)
                offset += len(Beams[idxinbatch])
                Beams[idxinbatch] = A
                min_len = min(min_len, min(len(hypo) for hypo in A))
            prefix_cache.pruneShorterThan(min_len)

            frame_lengths -= 1

        return [sorted(B_, key=lambda item: item.score, reverse=True)[
            :self.nbest] for B_ in Beams]

    def _lm_step(self, inp_tokens, hidden):
        """Forward a step of LM module, this equals to self.lm.forward() + log_softmax()"""
        logits, hs = self.lm(inp_tokens, hidden=hidden)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, hs


def collect_scores(hypos: List[Hypothesis], dummy_tensor: torch.Tensor = None) -> torch.Tensor:
    """Collect the scores of from hypos to a tensor"""
    if dummy_tensor is None:
        dummy_tensor = torch.empty(1)
    return dummy_tensor.new_tensor([hyp.score for hyp in hypos], dtype=torch.float)


def group_to_batch(hypos: List[Hypothesis], dummy_tensor: torch.Tensor = None, prefix_cache: PrefixCacheDict = None, statelen_fixed: bool = False) -> Tuple[List[int], torch.Tensor, Dict[str, AbsStates]]:
    """Group the hypothesis in the list into batch with their hidden states

    Args:
        hypos
        dummy_tensor : claim the device of created batches, if None, use cpu

    Returns:
        if prefix_cache=None:
            [(indices, batched_tokens, batched_states), ... ]
        else
            [(indices, batched_tokens, batched_states), ... ], [(indices, batched_output, batched_states), ...]
        indices (list(int)): index of hypo in the original input hypos after batching
        batched_tokens (torch.LongTensor): [N, 1]
        batched_states (Dict[str, AbsStates]): the hidden states of the hypotheses, depending on the prediction network type.
        batched_output (Dict[str, torch.Tensor]): the cached output being batched
        statelen_fixed (bool, default False): whether to group the states by hypo lengths, 
            if set True, this would slightly speedup training, however it requires the cache state to be of fixed length with variable seq lengths (like LSTM)
    """
    if dummy_tensor is None:
        dummy_tensor = torch.empty(1)

    hypos_with_index = list(enumerate(hypos))
    # split hypos into two groups, one with cache hit and the other the cache doesn't.
    if prefix_cache is not None:
        in_cache = []
        for id, hypo in hypos_with_index:
            if hypo.pred in prefix_cache:
                in_cache.append((id, hypo))
        for id, _ in in_cache[::-1]:
            hypos_with_index.pop(id)

    # group that cache doesn't hit
    batched_out = []
    if statelen_fixed:
        groups_uncached = [hypos_with_index] if len(
            hypos_with_index) > 0 else []
    else:
        groups_uncached = groupby(
            hypos_with_index, key=lambda item: len(item[1]))
    for _hypos_with_index in groups_uncached:
        _index, _hypos = list(zip(*_hypos_with_index))
        _batched_tokens = torch.cat(
            [hyp.pop_last() for hyp in _hypos], dim=0).view(-1, 1)
        _batched_states = {
            _key: _state.batching(
                [_hyp.cache[_key]for _hyp in _hypos]
            ) for _key, _state in _hypos[0].cache.items()
            if isinstance(_state, AbsStates)}     # type: Dict[str, AbsStates]

        batched_out.append((list(_index), _batched_tokens, _batched_states))

    if prefix_cache is None:
        return batched_out
    elif in_cache == []:
        return batched_out, []
    else:
        cached_out = []
        if statelen_fixed:
            groups_cached = [in_cache] if len(in_cache) > 0 else []
        else:
            groups_cached = groupby(in_cache, key=lambda item: len(item[1]))
        for _hypos_with_index in groups_cached:
            _index, _hypos = list(zip(*_hypos_with_index))
            # type: List[Dict[str, Union[torch.Tensor, AbsStates]]]
            caches = [prefix_cache.getCache(_hyp.pred) for _hyp in _hypos]
            _batched_out = {}
            _batched_states = {}
            for k in caches[0].keys():
                if isinstance(caches[0][k], AbsStates):
                    _batched_states[k] = caches[0][k].batching(
                        [_cache[k] for _cache in caches])
                else:
                    # [1, 1, H]
                    _batched_out[k] = torch.cat(
                        [_cache[k] for _cache in caches], dim=0)
            cached_out.append((list(_index), _batched_out, _batched_states))

        return batched_out, cached_out


def groupby(item_list: Iterable, key: Callable) -> List[List[Any]]:
    odict = {}  # type: Dict[Any, List]
    for item in item_list:
        _k = key(item)
        if _k not in odict:
            odict[_k] = [item]
        else:
            odict[_k].append(item)
    return list(odict.values())
