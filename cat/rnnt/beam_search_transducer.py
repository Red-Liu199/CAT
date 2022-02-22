"""Beam search for Transducer sequence.

Support:

- native decoding (with prefix tree merge)
- latency controlled decoding (with prefix tree merge)
- alignment-length synchronous decoding

Author: Huahuan Zhengh (maxwellzh@outlook.com)
"""

from .joint import AbsJointNet
from ..shared.decoder import AbsDecoder, AbsStates
from ..shared.data import sortedPadCollateLM
from ..shared import coreutils as utils

import os
import yaml
import pickle
import numpy as np
from typing import Dict, Union, List, Optional, Literal, Tuple, Any, Iterable, Callable

import torch


def fclone(fp: Union[float, torch.FloatTensor]):
    if isinstance(fp, float):
        return fp
    elif isinstance(fp, torch.Tensor):
        return fp.clone()
    else:
        raise NotImplementedError(f"Invalid type of float point {fp}")


class Hypothesis():
    def __init__(self,
                 pred: torch.LongTensor,
                 log_prob: Union[torch.Tensor, float],
                 cache: Union[Dict[str, Union[AbsStates, torch.Tensor]], None],
                 lm_score: Union[torch.Tensor, float] = 0.0,
                 len_norm: bool = False) -> None:

        self.pred = pred
        self.log_prob = log_prob
        self.cache = cache
        self._res_word = []
        self.lm_score = lm_score
        self.len_norm = len_norm

    @property
    def score(self):
        score = self.log_prob + self.lm_score
        if self.len_norm:
            return torch.as_tensor(score/len(self))
        else:
            return torch.as_tensor(score)

    def clone(self):
        new_hypo = Hypothesis(
            self.pred.clone(),
            fclone(self.log_prob),
            self.cache.copy(),
            fclone(self.lm_score),
            self.len_norm
        )
        new_hypo._res_word = self._res_word[:]
        return new_hypo

    def __add__(self, rhypo):
        new_hypo = self.clone()
        new_hypo.log_prob = torch.logaddexp(new_hypo.log_prob, rhypo.log_prob)
        return new_hypo

    def add_(self, rhypo):
        '''in-place version of __add__'''
        self.log_prob = torch.logaddexp(self.log_prob, rhypo.log_prob)
        return self

    def add_token(self, tok: torch.LongTensor):
        self.pred = torch.cat([self.pred, tok.view(1)])
        self._res_word.append(tok.item())

    def __len__(self) -> int:
        return len(self.pred)

    def __str__(self) -> str:
        return "Hypothesis({}, {:.2e})".format(self.pred[1:], (self.score if isinstance(self.score, float) else self.score.item()))

    def __repr__(self) -> str:
        return f"Hypothesis(pred={self.pred}, score={self.score:.2f})"


class PrefixTree():
    def __init__(self, _load_pth: str = None) -> None:
        self._tree = {}    # type: Dict[int, Dict]
        if _load_pth is not None:
            self.load(_load_pth)

    def update(self, pref: List[int], _word: str = None):
        tree = self._tree
        for k in pref:
            if k not in tree:
                tree[k] = {}
            tree = tree[k]
        tree[-1] = _word

    def load(self, pth: str):
        assert os.path.isfile(pth)
        with open(pth, 'rb') as fi:
            self._tree = pickle.load(fi)
        assert isinstance(self._tree, dict)

    def save(self, pth: str):
        with open(pth, 'wb') as fo:
            pickle.dump(self._tree, fo)
        return

    def havePref(self, prefix: List[int]) -> bool:
        '''A not strict version of __contains__ 
            __contains__ match whole word, 
            while havePref() only check prefix in vocab.
        '''
        tree = self._tree
        for k in prefix:
            if k not in tree:
                return False
            else:
                tree = tree[k]
        return True

    def __contains__(self, prefix: List[int]) -> bool:
        tree = self._tree
        for k in prefix:
            if k not in tree:
                return False
            else:
                tree = tree[k]
        if -1 in tree:
            return True
        else:
            return False

    def __sizeof__(self) -> int:
        return super().__sizeof__() + self._tree.__sizeof__()

    def __str__(self) -> str:
        return yaml.dump(self._tree, default_flow_style=False)

    def size(self) -> int:
        cnt = 0
        tocnt = [self._tree]    # type: List[Dict]
        while tocnt != []:
            t = tocnt.pop()
            if -1 in t:
                cnt += 1
            tocnt += [v for v in t.values() if isinstance(v, dict)]
        return cnt


class PrefixCacheDict():
    """
    This use a map-style way to store the cache.
    Compared to tree-like structure, thie would be less efficient when the tree is 
    quite large. But more efficient when it is small.
    """

    def __init__(self) -> None:
        self._cache = {}    # type: Dict[torch.LongTensor, Dict]

    def update(self, pref: torch.LongTensor, new_cache: dict):
        if pref in self._cache:
            self._cache[pref].update(new_cache.copy())
        else:
            self._cache[pref] = new_cache.copy()

    def getCache(self, pref: torch.LongTensor) -> Union[None, dict]:
        '''Get cache. If there isn't such prefix, return None.
        '''
        if pref in self._cache:
            return self._cache[pref]
        else:
            return None

    def pruneAllBut(self, legal_prefs: List[torch.LongTensor]):
        new_cache = {pref: self._cache[pref]
                     for pref in legal_prefs if pref in self._cache}
        del self._cache
        self._cache = new_cache

    def pruneShorterThan(self, L: int):
        torm = [key for key in self._cache if key.size(0) < L]
        for k in torm:
            del self._cache[k]

    def __str__(self) -> str:
        cache = {}
        for k in self._cache.keys():
            cache[k] = {}
            for _k in self._cache[k].keys():
                cache[k][_k] = 'cache'

        return yaml.dump(cache, default_flow_style=False)


def beam_append(ongoing_beams: List[Hypothesis], new_beam: Hypothesis, prefix_merge: bool = False) -> List[Hypothesis]:
    """Append the new hypothesis into ongoing_beams w/ or w/o prefix merging"""
    if prefix_merge:
        for _beam in ongoing_beams:
            if torch.equal(_beam.pred, new_beam.pred):
                _beam.add_(new_beam)
                return ongoing_beams

    ongoing_beams.append(new_beam)
    return ongoing_beams


def recombine_hypo(redundant_hypos: List[Hypothesis]) -> List[Hypothesis]:
    """Recombine the hypos to merge duplicate path"""
    out_hypos = []
    for hypo in redundant_hypos:
        out_hypos = beam_append(out_hypos, hypo, True)

    return out_hypos


class MaxBeamBuffer():
    """Maintain the max K hypo"""

    def __init__(self, buffer_size: int) -> None:
        assert buffer_size > 0
        self._buffer = {}   # type: Dict[int, Hypothesis]
        self._buffer_size = buffer_size
        self._min_index = -1    # int
        self._res = buffer_size
        self._iner_cnt = 0

    def reset(self):
        del self._buffer
        self._buffer = {}
        self._min_index = -1
        self._res = self._buffer_size
        self._iner_cnt = 0

    def update(self, hypo: Hypothesis):
        if self._res == self._buffer_size:
            self._buffer[self._iner_cnt] = hypo
            self._min_index = self._iner_cnt
            self._res -= 1
        elif self._res > 0:
            self._buffer[self._iner_cnt] = hypo
            self._res -= 1
            if hypo.score < self._buffer[self._min_index].score:
                self._min_index = self._iner_cnt
        else:
            if hypo.score > self._buffer[self._min_index].score:
                del self._buffer[self._min_index]
                self._buffer[self._iner_cnt] = hypo
                self._min_index = min(
                    self._buffer.keys(), key=lambda k: self._buffer[k].score)
        self._iner_cnt += 1

    def isfull(self) -> bool:
        return self._res == 0

    def min(self) -> Union[None, float, torch.Tensor]:
        if self._res > 0:
            return float('-inf')
        return self._buffer[self._min_index].score

    def getBeam(self) -> List[Hypothesis]:
        return list(self._buffer.values())

    def __repr__(self) -> str:
        return f"MaxBeamBuffer(maxsize={self._buffer_size}, ressize={self._res})"

    def __str__(self) -> str:
        return '\n'.join([f"MaxBeamBuffer({self._buffer_size}, {self._min_index})"] +
                         [f"{key} {hypo}" for key, hypo in self._buffer.items()])


class TransducerBeamSearcher():

    def __init__(
        self,
        decoder: AbsDecoder,
        joint: AbsJointNet,
        blank_id: int = 0,
        bos_id: int = 0,
        beam_size: int = 5,
        nbest: int = 1,
        algo: Literal['default', 'lc', 'alsd', 'rna'] = 'default',
        umax_portion: float = 0.35,  # for alsd, librispeech
        prefix_merge: bool = True,
        lm_module: Optional[AbsDecoder] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = 0.6,
        state_beam: float = 2.3,
        expand_beam: float = 2.3,
        temperature: float = 1.0,
        word_prefix_tree: Optional[str] = None,
        rescore: bool = False,
        verbose: bool = True
    ):
        super(TransducerBeamSearcher, self).__init__()
        assert blank_id == bos_id

        if alpha == 0.0:
            # disable lm fusion/rescore
            if verbose:
                print(f"WARNING: "
                      f"alpha = 0 will disable LM interation whatever beta is.")
            beta = None
            alpha = None
            lm_module = None

        self.decoder = decoder
        self.joint = joint
        self.blank_id = blank_id
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.nbest = min(nbest, beam_size)
        self.lm = lm_module
        self.alpha_ = alpha

        if lm_module is None and alpha is not None:
            raise ValueError("Language model is not provided.")

        self.is_latency_control = False
        self.is_prefix = prefix_merge
        self.rescore = rescore and (self.alpha_ is not None)
        self.beta_ = beta
        if self.rescore:
            # disable fusion
            self.alpha_ = None
            self.rescore_weight = alpha
        if algo == 'default':
            self.searcher = self.default_beam_search
        elif algo == 'lc':
            # latency controlled beam search
            self.is_latency_control = True
            self.state_beam = state_beam
            self.expand_beam = expand_beam
            self.searcher = self.default_beam_search
        elif algo == 'alsd':
            if word_prefix_tree is not None:
                self.word_prefix_tree = PrefixTree(word_prefix_tree)
            else:
                self.word_prefix_tree = None
            self.is_prefix = True
            self.u_portion = umax_portion
            assert umax_portion > 0.0 and umax_portion < 1.0
            self.searcher = self.align_length_sync_decode
        elif algo == 'rna':
            self.is_prefix = True,
            self.searcher = self.rna_decode
        else:
            raise RuntimeError(f"Unknown beam search algorithm: {algo}.")

        self.temp = temperature

    def __call__(self, tn_output):

        hyps = self.searcher(tn_output)     # type:List[Hypothesis]

        if self.rescore:
            rescored_index = self.call_rescore(hyps)
            hyps = [hyps[i] for i in rescored_index]

        pred_list, score_list = ([hypo.pred[1:] for hypo in hyps], [
            hypo.score for hypo in hyps])
        return pred_list, score_list

    def call_rescore(self, decoded_hypos: List[Hypothesis]):
        """Do rescoring with LM"""
        assert self.rescore and self.rescore_weight is not None, \
            f"Expect enable rescore=True with alpha specified, alpha={self.rescore_weight}"

        # [B, U]
        dummy_tensor = decoded_hypos[0].pred.new_empty(1)
        lens_in = dummy_tensor.new_tensor(
            [len(hypo) for hypo in decoded_hypos])
        in_seqs = utils.pad_list([hypo.pred for hypo in decoded_hypos], 0)

        # suppose </s> = <s>
        dummy_targets = torch.roll(in_seqs, -1, dims=1)
        log_lm_probs = self.lm.score(in_seqs, dummy_targets, lens_in)
        log_model_probs = collect_scores(
            decoded_hypos, dummy_tensor=dummy_tensor)

        composed_score = log_model_probs + self.rescore_weight * \
            log_lm_probs + self.beta_ * lens_in

        rescored_index = torch.argsort(composed_score, descending=True)
        return rescored_index

    def raw_decode(self, tn_output: torch.Tensor) -> List[Hypothesis]:
        return self.searcher(tn_output)     # type:List[Hypothesis]

    def default_beam_search(self, tn_output: torch.Tensor):
        """Transducer beam search decoder is a beam search decoder over batch which apply Transducer rules:
            1- for each utterance:
                2- for each time steps in the Transcription Network (TN) output:
                    -> Do forward on PN and Joint network
                    -> Select topK <= beam
                    -> Do a while loop extending the hyps until we reach blank
                        -> otherwise:
                        --> extend hyp by the new token

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [1, time_len, hiddens].
        """
        use_lm = self.alpha_ is not None
        # min between beam and max_target_lent
        tn_i = tn_output[0]
        dummy_tensor = torch.empty(
            (1, 1), device=tn_output.device, dtype=torch.long)
        blank = self.blank_id * torch.ones_like(dummy_tensor)

        input_PN = torch.ones_like(dummy_tensor) * self.bos_id
        # First forward-pass on PN
        hyp = Hypothesis(
            pred=dummy_tensor.new_tensor([self.bos_id]),
            log_prob=0.0,
            cache={'pn_state': None},
            len_norm=True
        )
        if use_lm:
            hyp.cache.update({"lm_state": None})

        beam_hyps = [hyp]
        prefix_cache = PrefixCacheDict()
        t_cache = {'pn_out': None, 'lm_out': None}

        # For each time step
        for tn_i_t in tn_i:
            prefix_cache.pruneAllBut([hypo.pred for hypo in beam_hyps])

            # get hyps for extension
            process_hyps = beam_hyps
            beam_hyps = []  # type: List[Hypothesis]
            while len(beam_hyps) < self.beam_size:
                # Add norm score
                a_best_hyp = max(
                    process_hyps, key=lambda x: x.score)  # type: Hypothesis

                # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                if self.is_latency_control and len(beam_hyps) > 0:
                    b_best_hyp = max(
                        beam_hyps, key=lambda x: x.score)
                    if b_best_hyp.score >= self.state_beam + a_best_hyp.score:
                        break

                # remove best hyp from process_hyps
                process_hyps.remove(a_best_hyp)
                t_best_pref = a_best_hyp.pred

                hypo_cache = prefix_cache.getCache(t_best_pref)
                if hypo_cache is None:
                    input_PN[0, 0] = t_best_pref[-1]
                    pn_out, hidden = self._pn_step(
                        input_PN, a_best_hyp.cache["pn_state"])
                    t_cache['pn_out'] = (pn_out, hidden)
                    if use_lm:
                        log_probs_lm, hidden_lm = self._lm_step(
                            input_PN, a_best_hyp.cache["lm_state"])
                        t_cache['lm_out'] = (log_probs_lm, hidden_lm)
                    prefix_cache.update(t_best_pref, t_cache)
                else:
                    pn_out, hidden = hypo_cache['pn_out']
                    if use_lm:
                        log_probs_lm, hidden_lm = hypo_cache['lm_out']

                # forward jointnet
                log_probs = self._joint_step(tn_i_t, pn_out)

                # Sort outputs at time
                logp_targets, tokens = torch.topk(
                    log_probs.view(-1), k=self.beam_size, dim=-1)

                if self.is_latency_control:
                    best_logp = (
                        logp_targets[0]
                        if tokens[0] != blank
                        else logp_targets[1])

                # Extend hyp by selection
                for log_p, tok in zip(logp_targets, tokens):
                    topk_hyp = a_best_hyp.clone()  # type: Hypothesis
                    topk_hyp.log_prob += log_p

                    if tok == self.blank_id:
                        # prune the beam with same prefix
                        beam_hyps = beam_append(
                            beam_hyps, topk_hyp, self.is_prefix)
                        continue
                    if (not self.is_latency_control) or (self.is_latency_control and log_p >= best_logp - self.expand_beam):
                        topk_hyp.add_token(tok)
                        topk_hyp.cache["pn_state"] = hidden
                        if use_lm:
                            topk_hyp.cache["lm_state"] = hidden_lm
                            topk_hyp.lm_score = self.alpha_ * \
                                log_probs_lm.view(-1)[tok] + self.beta_
                        process_hyps.append(topk_hyp)

        del prefix_cache
        nbest_hyps = sorted(
            beam_hyps,
            key=lambda x: x.score,
            reverse=True,
        )[: self.nbest]

        return nbest_hyps

    def align_length_sync_decode(self, tn_out: torch.Tensor):
        """
        "ALIGNMENT-LENGTH SYNCHRONOUS DECODING FOR RNN TRANSDUCER"

        tn_output: [1, T, D]
        """

        use_lm = self.alpha_ is not None
        use_wpt = self.word_prefix_tree is not None

        tn_out = tn_out[0]
        dummy_tensor = tn_out.new_empty(1, dtype=torch.long)
        B = [Hypothesis(
            pred=dummy_tensor.new_tensor([self.bos_id]),
            log_prob=0.0,
            cache={'pn_state': self.decoder.init_states()})]
        if use_lm:
            B[0].cache.update({'lm_state': self.lm.init_states()})
        F = []  # type: List[Hypothesis]
        T = tn_out.size(0)
        Umax = int(T * self.u_portion)
        prefix_cache = PrefixCacheDict()
        buffer = MaxBeamBuffer(self.beam_size)

        for i_path in range(T+Umax):
            buffer.reset()
            # remove the invalid hypos (t >= T) in the beam
            sliced_B = [hypo for hypo in B if i_path - len(hypo) + 1 < T]
            if sliced_B == []:
                break

            group_B_uncached, group_B_cached = group_to_batch(
                sliced_B, dummy_tensor, prefix_cache)
            i_sep = len(group_B_uncached)
            group_B = group_B_uncached + group_B_cached
            # e.g. [2, 3]
            n_group_batches = torch.LongTensor(
                [len(g[0]) for g in group_B])
            # e.g. [0, 0, 1, 1, 1]
            map_relidx2gid = torch.repeat_interleave(
                n_group_batches).tolist()
            # e.g. [0, 1, 0, 1, 2]
            map_relidx2bid = []
            for ng in n_group_batches:
                map_relidx2bid += list(range(ng))

            index_of_beams = []   # len: len(sliced_B)
            group_pn_out = []       # len: len(group_B)
            group_pn_state = []     # len: len(group_B)
            if use_lm:
                group_lm_out, group_lm_state = [], []
            for gid, (g_index, g_tokens, g_states) in enumerate(group_B):
                index_of_beams += g_index
                # [B_i, 1, H], ...
                if gid >= i_sep:
                    pn_out = g_tokens['pn_out']
                    pn_state = g_states['pn_state']()
                    if use_lm:
                        lm_out = g_tokens['lm_out']
                        lm_state = g_states['lm_state']()
                else:
                    pn_out, pn_state = self.decoder(
                        g_tokens,
                        g_states['pn_state']())
                    if use_lm:
                        lm_out, lm_state = self._lm_step(
                            g_tokens,
                            g_states['lm_state']())
                group_pn_out.append(pn_out)
                group_pn_state.append(pn_state)
                if use_lm:
                    group_lm_out.append(lm_out)
                    group_lm_state.append(lm_state)

            # joint net batching
            # time step for each group, len: len(group_B)
            group_t = [i_path - len(sliced_B[g[0][0]]) + 1 for g in group_B]
            # [B, 1, H]
            pn_out = torch.cat(group_pn_out, dim=0)
            # [B, 1, H]
            expand_tn_out = torch.cat([
                tn_out[t].expand(ng, 1, -1)
                for t, ng in zip(group_t, n_group_batches)], dim=0)
            # [B, 1, 1, V] -> [B, V]
            log_prob = (self.joint(expand_tn_out, pn_out)
                        ).squeeze(1).squeeze(1)
            # [B,]
            for i, _log_p in enumerate(log_prob[:, self.blank_id]):
                gid, rbid = map_relidx2gid[i], map_relidx2bid[i]
                cur_hypo = sliced_B[index_of_beams[i]].clone()
                cur_hypo.log_prob += _log_p
                buffer.update(cur_hypo)
                prefix_cache.update(cur_hypo.pred, {
                    'pn_out': group_pn_out[gid][rbid:rbid+1],
                    'pn_state': self.decoder.get_state_from_batch(group_pn_state[gid], rbid)
                })
                if use_lm:
                    prefix_cache.update(cur_hypo.pred, {
                        'lm_out': group_lm_out[gid][rbid:rbid+1],
                        'lm_state': self.lm.get_state_from_batch(group_lm_state[gid], rbid)
                    })
                if group_t[map_relidx2gid[i]] == T-1:
                    F.append(cur_hypo)

            if use_lm:
                lm_out = torch.cat(group_lm_out, dim=0)
                lm_score = self.alpha_ * \
                    lm_out.squeeze(1) + self.beta_
                # [B, V]
                fused_prob = log_prob + lm_score
            else:
                fused_prob = log_prob.clone()

            # mask the blank id postion, so that we won't select it in top-k hypos
            fused_prob[:, self.blank_id].fill_(float('-inf'))
            # calculate the hypo score here, then we can get the topk over all beams
            # [B, V] + [B, 1] -> [B, V]
            V = fused_prob.size(-1)
            fused_prob += collect_scores(
                [sliced_B[i] for i in index_of_beams],
                dummy_tensor).unsqueeze(1)

            # [K,]
            _, flatten_positions = torch.topk(
                fused_prob.flatten(), k=self.beam_size)

            index_batch = torch.div(
                flatten_positions, V, rounding_mode='floor')
            tokens = flatten_positions % V
            for gbid, tok in zip(index_batch, tokens):
                idx_g, idx_b_part = map_relidx2gid[gbid], map_relidx2bid[gbid]
                cur_hypo = sliced_B[index_of_beams[gbid]].clone()
                cur_hypo.add_token(tok)
                if use_wpt:
                    # if use word prefix tree, skip those not in the tree
                    if not self.word_prefix_tree.havePref(cur_hypo._res_word):
                        # emit the previous word
                        '''
                        NOTE (huahuan): because sentencepiece take space as token (or prefix of token), 
                        it's hard to tell whether there is a new word completed, unless introducing 
                        the sentencepiece model. e.g. (in pratical, words are token index, replaced to text for better understanding)
                            _res_word = ['_add', 'li'] -> '_addli' is not the prefix of any word ->
                                emit '_add' -> '_add' in prefix tree, so won't skip this hypo (unexpected)

                            _res_word = ['_add', '_li'] -> '_add_li' is not the prefix of any word ->
                                emit '_add' -> '_add' in prefix tree, so won't skip this hypo (as expected)
                        '''
                        complete_word = cur_hypo._res_word[:-1]
                        cur_hypo._res_word = cur_hypo._res_word[-1:]
                        if complete_word not in self.word_prefix_tree:
                            continue
                cur_hypo.log_prob += log_prob[gbid, tok]
                cur_hypo.cache['pn_state'] = self.decoder.get_state_from_batch(
                    group_pn_state[idx_g], idx_b_part)

                if use_lm:
                    cur_hypo.lm_score += lm_score[gbid, tok]
                    cur_hypo.cache['lm_state'] = self.lm.get_state_from_batch(
                        group_lm_state[idx_g], idx_b_part)
                buffer.update(cur_hypo)

            del B
            B = recombine_hypo(buffer.getBeam())
            prefix_cache.pruneShorterThan(i_path-max(group_t)+1)

        if F == []:
            F = B
        Nbest = sorted(F, key=lambda item: item.score,
                       reverse=True)[:self.nbest]

        return Nbest

    def batch_alsd(self, tn_out: torch.Tensor, lt: torch.LongTensor):
        """
        "ALIGNMENT-LENGTH SYNCHRONOUS DECODING FOR RNN TRANSDUCER"

        tn_output: [N, T, D]
        """

        use_lm = self.alpha_ is not None

        dummy_tensor = tn_out.new_empty(1, dtype=torch.long)
        N = tn_out.size(0)
        B = [[Hypothesis(
            pred=dummy_tensor.new_tensor([self.bos_id]),
            log_prob=0.0,
            cache={'pn_state': self.decoder.init_states()})] for _ in range(N)]
        if use_lm:
            for n in range(N):
                B[n][0].cache.update({'lm_state': self.lm.init_states()})
        Final = [[] for _ in range(N)]  # type: List[List[Hypothesis]]
        ly = (lt*self.u_portion).to(torch.long)
        Tmax = lt.max()
        Umax = ly.max()
        dummy_tensor = tn_out.new_empty(1)
        prefix_cache = PrefixCacheDict()
        buffers = [MaxBeamBuffer(self.beam_size) for _ in range(N)]

        for i_path in range(Tmax+Umax):

            sqz_beams = []  # type: List[Hypothesis]
            # remove the invalid hypos (t >= T) in the beam
            map_idxbeam2orin = []
            for n in range(N):
                buffers[n].reset()
                sliced_beams = [hypo for hypo in B[n]
                                if i_path - len(hypo) + 1 < lt[n] and i_path < lt[n] + ly[n]]
                sqz_beams += sliced_beams
                map_idxbeam2orin += [n] * len(sliced_beams)
            if sqz_beams == []:
                break

            group_uncached, group_cached = group_to_batch(
                sqz_beams, dummy_tensor, prefix_cache)
            i_sep = len(group_uncached)
            group_B = group_uncached + group_cached
            n_group_batches = torch.LongTensor(
                [len(g[0]) for g in group_B])
            num_batches = n_group_batches.sum()
            map_relidx2gid = torch.repeat_interleave(
                n_group_batches).tolist()
            map_relidx2bid = []
            for ng in n_group_batches:
                map_relidx2bid += list(range(ng))

            index_of_beams = []   # len: len(sliced_B)
            group_pn_out = []       # len: len(group_B)
            group_pn_state = []     # len: len(group_B)
            if use_lm:
                group_lm_out, group_lm_state = [], []
            for gid, (g_index, g_tokens, g_states) in enumerate(group_B):
                index_of_beams += g_index
                # [B_i, 1, H], ...
                if gid >= i_sep:
                    pn_out = g_tokens['pn_out']
                    pn_state = g_states['pn_state']()
                    if use_lm:
                        lm_out = g_tokens['lm_out']
                        lm_state = g_states['lm_state']()
                else:
                    pn_out, pn_state = self.decoder(
                        g_tokens,
                        g_states['pn_state']())
                    if use_lm:
                        lm_out, lm_state = self._lm_step(
                            g_tokens,
                            g_states['lm_state']())
                group_pn_out.append(pn_out)
                group_pn_state.append(pn_state)
                if use_lm:
                    group_lm_out.append(lm_out)
                    group_lm_state.append(lm_state)

            map_rel2orin = [map_idxbeam2orin[index_of_beams[i]]
                            for i in range(num_batches)]

            # joint net batching
            group_t = [i_path - len(sqz_beams[index_of_beams[i]]) +
                       1 for i in range(num_batches)]
            # [B, 1, H]
            pn_out = torch.cat(group_pn_out, dim=0)
            # [B, 1, H]
            expand_tn_out = torch.cat([tn_out[map_rel2orin[i], group_t[i]] for i in range(
                num_batches)], dim=0).view_as(pn_out)

            # [B, 1, 1, V] -> [B, V]
            log_prob = (self.joint(expand_tn_out, pn_out)
                        ).squeeze(1).squeeze(1)
            # [B,]
            for i, _log_p in enumerate(log_prob[:, self.blank_id]):
                cur_hypo = sqz_beams[index_of_beams[i]].clone()
                cur_hypo.log_prob += _log_p
                idx_orin = map_rel2orin[i]
                buffers[idx_orin].update(cur_hypo)
                gid, rbid = map_relidx2gid[i], map_relidx2bid[i]
                prefix_cache.update(cur_hypo.pred, {
                    'pn_out': group_pn_out[gid][rbid:rbid+1],
                    'pn_state': self.decoder.get_state_from_batch(group_pn_state[gid], rbid)
                })
                if use_lm:
                    prefix_cache.update(cur_hypo.pred, {
                        'lm_out': group_lm_out[gid][rbid:rbid+1],
                        'lm_state': self.lm.get_state_from_batch(group_lm_state[gid], rbid)
                    })
                if group_t[i] == lt[idx_orin]-1:
                    Final[idx_orin].append(cur_hypo)

            if use_lm:
                lm_out = torch.cat(group_lm_out, dim=0)
                lm_score = self.alpha_ * \
                    lm_out.squeeze(1) + self.beta_
                # [B, V]
                fused_prob = log_prob + lm_score
            else:
                fused_prob = log_prob.clone()

            # mask the blank id postion, so that we won't select it in top-k hypos
            fused_prob[:, self.blank_id].fill_(float('-inf'))
            # calculate the hypo score here, then we can get the topk over all beams
            # [B, V] + [B, 1] -> [B, V]
            fused_prob += collect_scores(
                [sqz_beams[i] for i in index_of_beams],
                dummy_tensor).unsqueeze(1)

            # [K,]
            scores, tokens = torch.topk(
                fused_prob, k=self.beam_size, dim=1)

            for b in range(num_batches):
                for k in range(self.beam_size):
                    idx_orin = map_rel2orin[b]
                    if scores[b, k] < buffers[idx_orin].min():
                        break
                    cur_hypo = sqz_beams[index_of_beams[b]].clone()
                    tok = tokens[b, k]
                    cur_hypo.add_token(tok)
                    cur_hypo.log_prob += log_prob[b, tok]
                    idx_g, idx_b_part = map_relidx2gid[b], map_relidx2bid[b]
                    cur_hypo.cache['pn_state'] = self.decoder.get_state_from_batch(
                        group_pn_state[idx_g], idx_b_part)

                    if use_lm:
                        cur_hypo.lm_score += lm_score[b, tok]
                        cur_hypo.cache['lm_state'] = self.lm.get_state_from_batch(
                            group_lm_state[idx_g], idx_b_part)

                    buffers[idx_orin].update(cur_hypo)
            del B
            B = [recombine_hypo(buffers[n].getBeam()) for n in range(N)]
            prefix_cache.pruneShorterThan(i_path-max(group_t)+1)

        batch_nbest = []
        for n in range(N):
            if Final[n] == []:
                Final[n] = B[n]
            Nbest = sorted(Final[n], key=lambda item: item.score,
                           reverse=True)[:self.nbest]
            batch_nbest.append(Nbest)

        return batch_nbest

    def rna_decode(self, tn_out: torch.Tensor) -> List[Hypothesis]:
        """
        One-step constrained decode for RNN-T or RNA decode

        tn_output: [1, T, D]        
        """
        use_lm = self.alpha_ is not None

        dummy_tensor = tn_out.new_empty(1, dtype=torch.long)
        B = [Hypothesis(
            pred=dummy_tensor.new_tensor([self.bos_id]),
            log_prob=0.0,
            cache={'pn_state': self.decoder.init_states()})]
        if use_lm:
            B[0].cache.update(
                {'lm_state': self.lm.init_states()})
        T = tn_out.size(1)
        prefix_cache = PrefixCacheDict()

        for t in range(T):
            nB = len(B)
            group_uncached, group_cached = group_to_batch(
                B, dummy_tensor, prefix_cache)
            group_beams = group_uncached + group_cached

            index_of_beams = []   # len: len(sliced_B)
            group_pn_out = []       # len: len(group_B)
            if use_lm:
                group_lm_out = []

            i_sep = len(group_uncached)
            for i, (g_index, g_tokens, g_states) in enumerate(group_beams):
                if i < i_sep:
                    index_of_beams += g_index
                    pn_out, pn_state = self.decoder(
                        g_tokens, g_states['pn_state']())
                    if use_lm:
                        lm_out, lm_state = self._lm_step(
                            g_tokens, g_states['lm_state']())
                    # add into cache
                    for bid, absidx in enumerate(g_index):
                        cur_cache = {
                            'pn_out': pn_out[bid:bid+1],
                            'pn_state': self.decoder.get_state_from_batch(pn_state, bid)
                        }
                        if use_lm:
                            cur_cache.update({
                                'lm_out': lm_out[bid:bid+1],
                                'lm_state': self.lm.get_state_from_batch(lm_state, bid)
                            })
                        prefix_cache.update(B[absidx].pred, cur_cache)
                else:
                    pn_out = g_tokens['pn_out']
                    pn_state = g_states['pn_state']()
                    if use_lm:
                        lm_out = g_tokens['lm_out']
                        lm_state = g_states['lm_state']()

                group_pn_out.append(pn_out)
                if use_lm:
                    group_lm_out.append(lm_out)

            # [uB, 1, H]
            pn_out = torch.cat(group_pn_out, dim=0)
            # [uB, 1, H]
            expand_tn_out = tn_out[:, t, :].expand(nB, -1, -1)
            log_prob = self.joint(
                expand_tn_out, pn_out).squeeze(1).squeeze(1)

            if use_lm:
                lm_out = torch.cat(group_lm_out, dim=0)
                lm_score = self.alpha_ * \
                    lm_out.squeeze(1) + self.beta_
                # [B, V]
                combine_score = log_prob + lm_score
                combine_score[:, self.blank_id] = log_prob[:, self.blank_id]
            else:
                combine_score = log_prob.clone()

            V = combine_score.size(-1)
            combine_score += collect_scores([B[i] for i in index_of_beams],
                                            dummy_tensor).unsqueeze(1)
            # [K, ]
            _, flatten_positions = torch.topk(
                combine_score.flatten(), k=self.beam_size)
            idx_beam = torch.div(
                flatten_positions, V, rounding_mode='floor')
            tokens = flatten_positions % V
            # [K, ]
            A = []
            for i, b in enumerate(idx_beam):
                orin_hypo = B[index_of_beams[b]]
                cur_hypo = orin_hypo.clone()
                cur_hypo.log_prob += log_prob[b, tokens[i]]
                if tokens[i] == self.blank_id:
                    A.append(cur_hypo)
                    continue
                cur_hypo.add_token(tokens[i])
                cur_hypo.cache = prefix_cache.getCache(orin_hypo.pred)
                if use_lm:
                    cur_hypo.lm_score += lm_score[b, tokens[i]]
                A.append(cur_hypo)

            B = recombine_hypo(A)
            prefix_cache.pruneShorterThan(min(len(hypo) for hypo in B))

        Nbest = sorted(B, key=lambda item: item.score,
                       reverse=True)[:self.nbest]

        return Nbest

    def _joint_step(self, tn_out: torch.Tensor, pn_out: torch.Tensor):
        """Join predictions (TN & PN)."""

        tn_out = tn_out.view(-1)
        pn_out = pn_out.view(-1)

        if self.temp == 1.0:
            return self.joint(tn_out, pn_out)
        else:
            logits = self.joint.impl_forward(
                tn_out, pn_out)
            return torch.log_softmax(logits/self.temp, dim=-1)

    def _lm_step(self, inp_tokens, memory):
        logits, hs = self.lm(inp_tokens, hidden=memory)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, hs

    def _pn_step(self, input_PN, hidden=None):

        return self.decoder(input_PN, hidden)


def collect_scores(hypos: List[Hypothesis], dummy_tensor: torch.Tensor = None) -> torch.Tensor:
    """Collect the scores of from hypos to a tensor"""
    if dummy_tensor is None:
        dummy_tensor = torch.empty(1)
    return dummy_tensor.new_tensor([hyp.score for hyp in hypos], dtype=torch.float)


def group_to_batch(hypos: List[Hypothesis], dummy_tensor: torch.Tensor = None, prefix_cache: PrefixCacheDict = None) -> Tuple[List[int], torch.Tensor, Dict[str, AbsStates]]:
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
    """
    if dummy_tensor is None:
        dummy_tensor = torch.empty(1)

    hypos_with_index = list(enumerate(hypos))
    # split hypos into two groups, one with cache hit and the other the cache doesn't.
    if prefix_cache is not None:
        in_cache = []
        for id, hypo in hypos_with_index:
            trycache = prefix_cache.getCache(hypo.pred)
            if trycache is None:
                continue
            in_cache.append((id, hypo))
        for id, _ in in_cache[::-1]:
            hypos_with_index.pop(id)

    # group that cache doesn't hit
    batched_out = []
    for _hypos_with_index in groupby(hypos_with_index, key=lambda item: item[1].pred.size(0)):
        _index, _hypos = list(zip(*_hypos_with_index))
        _batched_tokens = torch.stack(
            [hyp.pred[-1] for hyp in _hypos]).view(-1, 1)
        _batched_states = {_key:
                           _state.batching([_hyp.cache[_key]
                                           for _hyp in _hypos])
                           for _key, _state in _hypos[0].cache.items()
                           if isinstance(_state, AbsStates)}     # type: Dict[str, AbsStates]

        batched_out.append((list(_index), _batched_tokens, _batched_states))

    if prefix_cache is None:
        return batched_out
    elif in_cache == []:
        return batched_out, []
    else:
        cached_out = []
        for _hypos_with_index in groupby(in_cache, key=lambda item: item[1].pred.size(0)):
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
