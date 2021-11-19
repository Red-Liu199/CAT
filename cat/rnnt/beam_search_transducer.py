"""Beam search for Transducer sequence.

Support:

- native decoding (with prefix tree merge)
- latency controlled decoding (with prefix tree merge)
- alignment-length synchronous decoding

Author: Huahuan Zhengh (maxwellzh@outlook.com)
"""

from . import JointNet
from ..shared.decoder import AbsDecoder, AbsStates
from ..shared.data import sortedPadCollateLM

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
                 pred: List[int],
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
            return score/len(self)
        else:
            return score

    def clone(self):
        new_hypo = Hypothesis(
            self.pred[:],
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

    def add_token(self, tok: int):
        self.pred.append(tok)
        self._res_word.append(tok)

    def __len__(self) -> int:
        return len(self.pred)

    def __str__(self) -> str:
        return "Hypothesis({}, {:.2e})".format(self.pred[1:], (self.score if isinstance(self.score, float) else self.score.item()))


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
        self._cache = {}    # type: Dict[str, Dict]

    def update(self, pref: List[int], new_cache: dict):
        map_pref = ' '.join(map(str, pref))

        if map_pref in self._cache:
            self._cache[map_pref].update(new_cache.copy())
        else:
            self._cache[map_pref] = new_cache.copy()

    def getCache(self, pref: List[int]) -> Union[None, dict]:
        '''Get cache. If there isn't such prefix, return None.
        '''
        map_pref = ' '.join(map(str, pref))
        if map_pref in self._cache:
            return self._cache[map_pref]
        else:
            return None

    def pruneAllBut(self, legal_prefs: List[List[int]]):
        legal_maps = [' '.join(map(str, pref)) for pref in legal_prefs]

        new_cache = {pref: self._cache[pref]
                     for pref in legal_maps if pref in self._cache}
        del self._cache
        self._cache = new_cache

    def pruneShorterThan(self, L: int):
        torm = [key for key in self._cache if len(key) < 2*L-1]
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
            if _beam.pred == new_beam.pred:
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

    def getBeam(self) -> List[Hypothesis]:
        return list(self._buffer.values())

    def __str__(self) -> str:
        return '\n'.join([f"MaxBeamBuffer({self._buffer_size}, {self._min_index})"] +
                         [f"{key} {hypo}" for key, hypo in self._buffer.items()])


class TransducerBeamSearcher(torch.nn.Module):

    def __init__(
        self,
        decoder: AbsDecoder,
        joint: JointNet,
        blank_id: int = 0,
        bos_id: int = 0,
        beam_size: int = 5,
        nbest: int = 1,
        algo: Literal['default', 'lc', 'alsd', 'osc'] = 'default',
        umax_portion: float = 0.35,  # for alsd, librispeech
        prefix_merge: bool = True,
        lm_module: Optional[AbsDecoder] = None,
        lm_weight: float = 0.0,
        state_beam: float = 2.3,
        expand_beam: float = 2.3,
        temperature: float = 1.0,
        word_prefix_tree: Optional[str] = None,
        rescore: bool = False
    ):
        super(TransducerBeamSearcher, self).__init__()
        assert blank_id == bos_id
        self.decoder = decoder
        self.joint = joint
        self.blank_id = blank_id
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.nbest = min(nbest, beam_size)
        self.lm = lm_module
        self.lm_weight = lm_weight

        if lm_module is None and lm_weight > 0.0:
            raise ValueError("Language model is not provided.")

        self.is_latency_control = False
        self.is_prefix = prefix_merge
        self.rescore = rescore and self.lm_weight > 0.0
        if self.rescore:
            # disable fusion
            self.lm_weight = 0.0
            self.rescore_weight = lm_weight
        if algo == 'default':
            self.searcher = self.native_beam_search
        elif algo == 'lc':
            # latency controlled beam search
            self.is_latency_control = True
            self.state_beam = state_beam
            self.expand_beam = expand_beam
            self.searcher = self.native_beam_search
        elif algo == 'alsd':
            if word_prefix_tree is not None:
                self.word_prefix_tree = PrefixTree(word_prefix_tree)
            else:
                self.word_prefix_tree = None
            self.is_prefix = True
            self.u_portion = umax_portion
            assert umax_portion > 0.0 and umax_portion < 1.0
            self.searcher = self.align_length_sync_decode
        elif algo == 'osc':
            self.is_prefix = True
            self.searcher = self.one_step_sync_decode
        else:
            raise RuntimeError(f"Unknown beam search algorithm: {algo}.")

        self.temp = temperature

    def forward(self, tn_output):

        hyps = self.searcher(tn_output)

        if self.rescore:
            pred_list, score_list = hyps[0][0], hyps[1][0]
            collate_fn = sortedPadCollateLM(flatten_target=False)
            tokens, token_length, target, _ = collate_fn(
                [torch.LongTensor(p) for p in pred_list])
            logits, _ = self.lm(tokens, input_lengths=token_length)
            log_prob_lm = logits.log_softmax(
                dim=-1).gather(index=target.unsqueeze(2), dim=-1).squeeze(-1)
            final_scores = []
            for i in range(len(pred_list)):
                final_scores.append((score_list[i] + self.lm_weight * log_prob_lm[i,
                                    :token_length[i]].mean()))  # + beta_l*token_length[i]))
            pair = sorted(list(zip(pred_list, final_scores)),
                          key=lambda item: item[1], reverse=True)
            return list(zip(*pair))
        else:
            return hyps

    def native_beam_search(self, tn_output: torch.Tensor):
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
            [batch, time_len, hiddens].
        """

        # min between beam and max_target_lent
        nbest_batch = []
        nbest_batch_score = []
        dummy_tensor = torch.empty(
            (1, 1), device=tn_output.device, dtype=torch.int32)
        blank = self.blank_id * torch.ones_like(dummy_tensor)

        for tn_i in tn_output:
            input_PN = torch.ones_like(dummy_tensor) * self.bos_id
            # First forward-pass on PN
            hyp = Hypothesis(
                pred=[self.bos_id],
                log_prob=0.0,
                cache={'pn_state': None},
                len_norm=True
            )
            if self.lm_weight > 0:
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
                        if self.lm_weight > 0:
                            log_probs_lm, hidden_lm = self._lm_step(
                                input_PN, a_best_hyp.cache["lm_state"])
                            t_cache['lm_out'] = (log_probs_lm, hidden_lm)
                        prefix_cache.update(t_best_pref, t_cache)
                    else:
                        pn_out, hidden = hypo_cache['pn_out']
                        if self.lm_weight > 0:
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
                            topk_hyp.add_token(tok.item())
                            topk_hyp.cache["pn_state"] = hidden
                            if self.lm_weight > 0.0:
                                topk_hyp.cache["lm_state"] = hidden_lm
                                topk_hyp.lm_score = self.lm_weight * \
                                    log_probs_lm.view(-1)[tok]
                            process_hyps.append(topk_hyp)

            del prefix_cache
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x.score,
                reverse=True,
            )[: self.nbest]
            nbest_batch.append([hyp.pred[1:] for hyp in nbest_hyps])
            nbest_batch_score.append([hyp.score for hyp in nbest_hyps])

        return (nbest_batch, nbest_batch_score)

    def align_length_sync_decode(self, tn_out: torch.Tensor):
        """
        "ALIGNMENT-LENGTH SYNCHRONOUS DECODING FOR RNN TRANSDUCER"

        tn_output: [1, T, D]
        """

        use_lm = self.lm_weight > 0.0
        use_wpt = self.word_prefix_tree is not None
        beta_l = 0.6

        tn_out = tn_out[0]
        B = [Hypothesis(
            pred=[self.bos_id],
            log_prob=0.0,
            cache={'pn_state': self.decoder.init_states()})]
        if use_lm:
            B[0].cache.update({'lm_state': self.lm.init_states()})
        F = []  # type: List[Hypothesis]
        T = tn_out.size(0)
        Umax = int(T * self.u_portion)
        dummy_tensor = tn_out.new_empty(1)
        prefix_cache = PrefixCacheDict()

        for i_path in range(T+Umax):
            buffer = MaxBeamBuffer(self.beam_size)
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
                    # add cache
                    for i, _gid in enumerate(g_index):
                        c_pred = sliced_B[_gid].pred
                        prefix_cache.update(c_pred, {
                            'pn_out': pn_out[i:i+1],
                            'pn_state': self.decoder.get_state_from_batch(pn_state, i)
                        })
                        if use_lm:
                            prefix_cache.update(c_pred, {
                                'lm_out': lm_out[i:i+1],
                                'lm_state': self.lm.get_state_from_batch(lm_state, i)
                            })
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
                tn_out[t].expand(ng, 1, tn_out.size(-1))
                for t, ng in zip(group_t, n_group_batches)], dim=0)
            # [B, 1, 1, V] -> [B, V]
            log_prob = (self.joint(expand_tn_out, pn_out)
                        ).squeeze(1).squeeze(1)
            # [B,]
            for i, _log_p in enumerate(log_prob[:, self.blank_id]):
                cur_hypo = sliced_B[index_of_beams[i]].clone()
                cur_hypo.log_prob += _log_p
                buffer.update(cur_hypo)
                if group_t[map_relidx2gid[i]] == T-1:
                    F.append(cur_hypo)

            if use_lm:
                lm_out = torch.cat(group_lm_out, dim=0)
                lm_score = self.lm_weight * \
                    lm_out.squeeze(1) + beta_l
                # [B, V]
                fused_prob = log_prob + lm_score
            else:
                fused_prob = log_prob.clone()

            # mask the blank id postion, so that we won't select it in top-k hypos
            fused_prob[:, self.blank_id].fill_(float('-inf'))
            # calculate the hypo score here, then we can get the topk over all beams
            # [B, V] + [B, 1] -> [B, V]
            fused_prob += collect_scores(
                [sliced_B[i] for i in index_of_beams],
                dummy_tensor).unsqueeze(1)

            # [K,]
            _, flatten_positions = torch.topk(
                fused_prob.flatten(), k=self.beam_size)
            paired_index = np.array(np.unravel_index(
                flatten_positions.numpy(), fused_prob.shape)).T
            for gbid, tok in paired_index:
                idx_g, idx_b_part = map_relidx2gid[gbid], map_relidx2bid[gbid]
                cur_hypo = sliced_B[index_of_beams[gbid]].clone()
                cur_hypo.add_token(tok.item())
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

            B = recombine_hypo(buffer.getBeam())
            prefix_cache.pruneShorterThan(min([len(hypo) for hypo in B]))

        if F == []:
            F = B
        Nbest = sorted(F, key=lambda item: item.score,
                       reverse=True)[:self.nbest]
        pred_list, score_list = [hypo.pred[1:]
                                 for hypo in Nbest], [hypo.score for hypo in Nbest]

        return [pred_list], [score_list]

    def one_step_sync_decode(self, tn_out: torch.Tensor):

        use_lm = self.lm_weight > 0.0
        beta_l = 0.6

        tn_out = tn_out[0]
        B = [Hypothesis(
            pred=[self.bos_id],
            log_prob=0.0,
            cache={'pn_state': self.decoder.init_states()},
            len_norm=not use_lm)]

        T = tn_out.size(0)
        dummy_tensor = tn_out.new_empty(1)
        cache_pn = PrefixCacheDict()
        if use_lm:
            B[0].cache.update({'lm_state': self.lm.init_states()})
            cache_lm = PrefixCacheDict()

        for t in range(T):
            buffer = MaxBeamBuffer(self.beam_size)
            A_emit = B

            for step in range(2):
                group_B_uncached, group_B_cached = group_to_batch(
                    A_emit, dummy_tensor, cache_pn)
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
                for g_index, g_tokens, g_states in group_B[:i_sep]:
                    index_of_beams += g_index
                    pn_out, pn_state = self.decoder(
                        g_tokens,
                        g_states['pn_state']())
                    group_pn_out.append(pn_out)
                    group_pn_state.append(pn_state)
                    # add cache
                    if step == 1 and use_lm:
                        continue
                    for i, _gid in enumerate(g_index):
                        c_pred = A_emit[_gid].pred
                        cache_pn.update(c_pred, {
                            'pn_out': pn_out[i:i+1],
                            'pn_state': self.decoder.get_state_from_batch(pn_state, i)
                        })
                for g_index, g_out, g_states in group_B[i_sep:]:
                    index_of_beams += g_index
                    # [B_i, 1, H], ...
                    pn_out = g_out['pn_out']
                    pn_state = g_states['pn_state']()
                    group_pn_out.append(pn_out)
                    group_pn_state.append(pn_state)

                # joint net batching
                # [B, 1, H]
                pn_out = torch.cat(group_pn_out, dim=0)
                # [B, 1, H]
                expand_tn_out = tn_out[t].expand(
                    pn_out.size(0), 1, tn_out.size(-1))
                # [B, 1, 1, V] -> [B, V]
                log_prob = (self.joint(expand_tn_out, pn_out)
                            ).squeeze(1).squeeze(1)
                # [B,]
                for i, _log_p in enumerate(log_prob[:, self.blank_id]):
                    cur_hypo = A_emit[index_of_beams[i]].clone()
                    cur_hypo.log_prob += _log_p
                    buffer.update(cur_hypo)
                if step >= 1:
                    break

                if use_lm:
                    group_B_uncached, group_B_cached = group_to_batch(
                        A_emit, dummy_tensor, cache_lm)
                    i_sep = len(group_B_uncached)
                    group_B = group_B_uncached + group_B_cached
                    group_lm_out, group_lm_state = [], []
                    for g_index, g_tokens, g_states in group_B[:i_sep]:
                        lm_out, lm_state = self._lm_step(
                            g_tokens,
                            g_states['lm_state']())
                        group_lm_out.append(lm_out)
                        group_lm_state.append(lm_state)
                        # add cache
                        for i, _gid in enumerate(g_index):
                            c_pred = A_emit[_gid].pred
                            cache_lm.update(c_pred, {
                                'lm_out': lm_out[i:i+1],
                                'lm_state': self.lm.get_state_from_batch(lm_state, i)
                            })
                    for g_index, g_out, g_states in group_B[i_sep:]:
                        lm_out = g_out['lm_out']
                        lm_state = g_states['lm_state']()
                        group_lm_out.append(lm_out)
                        group_lm_state.append(lm_state)

                    lm_out = torch.cat(group_lm_out, dim=0)
                    lm_score = self.lm_weight * lm_out.squeeze(1) + beta_l
                    # [B, V]
                    fused_prob = log_prob + lm_score
                else:
                    fused_prob = log_prob.clone()

                # calculate the hypo score here, then we can get the topk over all beams
                # [B, V] + [B, 1] -> [B, V]
                fused_prob += collect_scores(
                    [A_emit[i] for i in index_of_beams],
                    dummy_tensor).unsqueeze(1)
                # mask the blank id postion, so that we won't select it in top-k hypos
                fused_prob[:, self.blank_id].fill_(float('-inf'))

                # [K,]
                A_emit_tmp = []
                _, flatten_positions = torch.topk(
                    fused_prob.flatten(), k=self.beam_size)
                paired_index = np.array(np.unravel_index(
                    flatten_positions.numpy(), fused_prob.shape)).T
                for gbid, tok in paired_index:
                    idx_g, idx_b_part = map_relidx2gid[gbid], map_relidx2bid[gbid]
                    cur_hypo = A_emit[index_of_beams[gbid]].clone()
                    cur_hypo.add_token(tok.item())
                    cur_hypo.log_prob += log_prob[gbid, tok]
                    cur_hypo.cache['pn_state'] = self.decoder.get_state_from_batch(
                        group_pn_state[idx_g], idx_b_part)
                    if use_lm:
                        cur_hypo.lm_score += lm_score[gbid, tok]
                        cur_hypo.cache['lm_state'] = self.lm.get_state_from_batch(
                            group_lm_state[idx_g], idx_b_part)
                    A_emit_tmp.append(cur_hypo)
                A_emit = A_emit_tmp

            B = recombine_hypo(buffer.getBeam())
            cache_pn.pruneShorterThan(min([len(hypo) for hypo in B]))
            if use_lm:
                cache_lm.pruneShorterThan(min([len(hypo) for hypo in B]))

        Nbest = sorted(B, key=lambda item: item.score,
                       reverse=True)[:self.nbest]

        return ([[hypo.pred[1:] for hypo in Nbest]], [[hypo.score for hypo in Nbest]])

    def _joint_step(self, tn_out: torch.Tensor, pn_out: torch.Tensor):
        """Join predictions (TN & PN)."""

        tn_out = tn_out.view(-1)
        pn_out = pn_out.view(-1)

        if self.temp == 1.0:
            return self.joint(tn_out, pn_out)
        else:
            logits = self.joint.skip_softmax_forward(
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
    for _hypos_with_index in groupby(hypos_with_index, key=lambda item: len(item[1].pred)):
        _index, _hypos = list(zip(*_hypos_with_index))
        _batched_tokens = dummy_tensor.new_tensor(
            [hyp.pred[-1] for hyp in _hypos], dtype=torch.long).view(-1, 1)
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
        for _hypos_with_index in groupby(in_cache, key=lambda item: len(item[1].pred)):
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
