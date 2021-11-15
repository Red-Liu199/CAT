"""Beam search for Transducer sequence.

Support:

- native decoding (with prefix tree merge)
- latency controlled decoding (with prefix tree merge)
- alignment-length synchronous decoding

Author: Huahuan Zhengh (maxwellzh@outlook.com)
"""

from . import JointNet
from ..shared.decoder import AbsDecoder, AbsStates

import copy
import yaml
from typing import Dict, Union, List, Optional, Literal, Tuple, Any, Iterable, Callable

import torch


class Hypothesis():
    def __init__(self, pred: List[int], score: Union[torch.Tensor, float], cache: Union[Dict[str, Union[AbsStates, torch.Tensor]], None]) -> None:
        self.pred = pred
        self.score = score
        self.cache = cache

    def clone(self):
        new_hypo = Hypothesis(
            self.pred[:],
            self.score if isinstance(
                self.score, float) else self.score.clone(),
            self.cache.copy()
        )
        return new_hypo

    def __len__(self) -> int:
        return len(self.pred)

    def __str__(self) -> str:
        return "Hypothesis({}, {:.2e})".format(self.pred[1:], (self.score if isinstance(self.score, float) else self.score.item()))


class PrefixCacheTree():
    def __init__(self) -> None:
        self._cache = {}    # type: Dict[int, Dict]

    def update(self, pref: List[int], new_cache: dict):
        tree = self._cache
        for k in pref:
            if k not in tree:
                tree[k] = {}
            tree = tree[k]

        if -1 in tree:
            print("Prefix:", pref)
            print(self)
            raise RuntimeError("Trying to update existing state.")
        else:
            tree[-1] = new_cache.copy()

    def getCache(self, pref: List[int]) -> Union[None, dict]:
        '''Get cache. If there isn't such prefix, return None.
        '''
        tree = self._cache
        for k in pref:
            if k not in tree:
                return None
            else:
                tree = tree[k]

        if -1 not in tree:
            return None
        else:
            return tree[-1]

    def pruneAllBut(self, legal_prefs: List[List[int]]):
        legal_cache = []
        for pref in legal_prefs:
            cache = self.getCache(pref)
            if cache is not None:
                legal_cache.append((pref, cache))
        del self._cache
        self._cache = {}
        for pref, cache in legal_cache:
            self.update(pref, cache)

    def __str__(self) -> str:
        cache = copy.deepcopy(self._cache)
        todeal = [cache]
        while todeal != []:
            tree = todeal.pop()
            for k in tree:
                if k == -1:
                    tree[k] = 'cache'
                else:
                    todeal.append(tree[k])

        return yaml.dump(cache, default_flow_style=False)


class PrefixCacheDict():
    """
    This use a map-style way to store the cache.
    Compared to PrefixCacheTree, thie would be less efficient when the tree is 
    quite large. But more efficient when it is small.
    """

    def __init__(self) -> None:
        self._cache = {}    # type: Dict[int, Dict]

    def update(self, pref: List[int], new_cache: dict):
        map_pref = ' '.join(map(str, pref))

        if map_pref in self._cache:
            return
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
            cache[k] = 'cache'

        return yaml.dump(cache, default_flow_style=False)


def beam_append(ongoing_beams: List[Hypothesis], new_beam: Hypothesis, prefix_merge: bool = False) -> List[Hypothesis]:
    """Append the new hypothesis into ongoing_beams w/ or w/o prefix merging"""
    if prefix_merge:
        for _beam in ongoing_beams:
            if _beam.pred == new_beam.pred:
                _beam.score = torch.logaddexp(
                    _beam.score, new_beam.score)
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
        algo: Literal['default', 'lc', 'alsd'] = 'default',
        umax_portion: float = 0.35,  # for alsd, librispeech
        prefix_merge: bool = True,
        lm_module: Optional[AbsDecoder] = None,
        lm_weight: float = 0.0,
        state_beam: float = 2.3,
        expand_beam: float = 2.3,
        temperature: float = 1.0
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
        if algo == 'default':
            self.searcher = self.native_beam_search
        elif algo == 'lc':
            # latency controlled beam search
            self.is_latency_control = True
            self.state_beam = state_beam
            self.expand_beam = expand_beam
            self.searcher = self.native_beam_search
        elif algo == 'alsd':
            self.is_prefix = True
            self.u_portion = umax_portion
            assert umax_portion > 0.0 and umax_portion < 1.0
            self.searcher = self.align_length_sync_decode
        else:
            raise RuntimeError(f"Unknown beam search algorithm: {algo}.")

        self.temp = temperature

    def forward(self, tn_output):
        """
        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        Topk hypotheses
        """

        hyps = self.searcher(tn_output)
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
                score=0.0,
                cache={'dec': None}
            )
            if self.lm_weight > 0:
                hyp.cache.update({"lm": None})

            beam_hyps = [hyp]
            prefix_cache = PrefixCacheDict()  # PrefixCacheTree()
            t_cache = {'out_pn': None, 'out_lm': None}

            # For each time step
            for tn_i_t in tn_i:
                prefix_cache.pruneAllBut([hypo.pred for hypo in beam_hyps])

                # get hyps for extension
                process_hyps = beam_hyps
                beam_hyps = []  # type: List[Hypothesis]
                while len(beam_hyps) < self.beam_size:
                    # Add norm score
                    a_best_hyp = max(
                        process_hyps, key=lambda x: x.score / len(x))  # type: Hypothesis

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if self.is_latency_control and len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps, key=lambda x: x.score / len(x))
                        a_best_prob = a_best_hyp.score
                        b_best_prob = b_best_hyp.score
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    # remove best hyp from process_hyps
                    process_hyps.remove(a_best_hyp)
                    t_best_pref = a_best_hyp.pred

                    hypo_cache = prefix_cache.getCache(t_best_pref)
                    if hypo_cache is None:
                        input_PN[0, 0] = t_best_pref[-1]
                        pn_out, hidden = self._pn_step(
                            input_PN, a_best_hyp.cache["dec"])
                        t_cache['out_pn'] = (pn_out, hidden)
                        if self.lm_weight > 0:
                            log_probs_lm, hidden_lm = self._lm_step(
                                input_PN, a_best_hyp.cache["lm"])
                            t_cache['out_lm'] = (log_probs_lm, hidden_lm)
                        prefix_cache.update(t_best_pref, t_cache)
                    else:
                        pn_out, hidden = hypo_cache['out_pn']
                        if self.lm_weight > 0:
                            log_probs_lm, hidden_lm = hypo_cache['out_lm']

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
                        topk_hyp.score += log_p

                        if tok == self.blank_id:
                            # prune the beam with same prefix
                            beam_hyps = beam_append(
                                beam_hyps, topk_hyp, self.is_prefix)
                            continue
                        if (not self.is_latency_control) or (self.is_latency_control and log_p >= best_logp - self.expand_beam):
                            topk_hyp.pred.append(tok.item())
                            topk_hyp.cache["dec"] = hidden
                            if self.lm_weight > 0.0:
                                topk_hyp.cache["lm"] = hidden_lm
                                topk_hyp.score += self.lm_weight * \
                                    log_probs_lm.view(-1)[tok]
                            process_hyps.append(topk_hyp)

            del prefix_cache
            # Add norm score
            for b in beam_hyps:
                b.score /= len(b)

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

        tn_out = tn_out[0]
        B = [Hypothesis(
            pred=[self.bos_id],
            score=0.0,
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

            # group B according to len(hypo.pred)
            group_B_uncached, group_B_cached = group_to_batch(
                sliced_B, dummy_tensor, prefix_cache)
            sep_i = len(group_B_uncached)
            group_B = group_B_uncached + group_B_cached

            original_indices = []
            group_pn_out = []
            group_pn_state = []
            group_tn = []
            group_t = []
            map_rel_index_to_groupid = {}
            map_rel_index_to_batchid = {}
            if use_lm:
                group_lm_out = []
                group_lm_state = []
            for gid, (g_index, g_tokens, g_states) in enumerate(group_B):
                n_g_batch = len(g_index)
                u = len(sliced_B[g_index[0]]) - 1
                t = i_path - u
                for bid, rel_i in enumerate(range(len(original_indices), len(original_indices)+n_g_batch)):
                    map_rel_index_to_groupid[rel_i] = gid
                    map_rel_index_to_batchid[rel_i] = bid

                group_t.append(t)
                # [H] -> [B_i, 1, H]
                group_tn.append(tn_out[t].expand(
                    n_g_batch, 1, tn_out.size(-1)))
                original_indices += g_index
                # decoder batching
                # [B_i, 1, H], ...
                if gid >= sep_i:
                    g_out = g_tokens    # type: Dict[str, torch.Tensor]
                    pn_out = g_out['pn_out']
                    hidden = g_states['pn_state']()
                else:
                    pn_out, hidden = self.decoder(
                        g_tokens, hidden=g_states['pn_state']())
                group_pn_out.append(pn_out)
                group_pn_state.append(hidden)
                if use_lm:
                    if gid >= sep_i:
                        lm_out = g_out['lm_out']
                        lm_hidden = g_states['lm_state']()
                    else:
                        lm_out, lm_hidden = self._lm_step(
                            g_tokens, g_states['lm_state']())
                    group_lm_out.append(lm_out)
                    group_lm_state.append(lm_hidden)

            # joint net batching
            # [B, 1, H]
            pn_out = torch.cat(group_pn_out, dim=0)
            # [B, 1, H]
            expand_tn_out = torch.cat(group_tn, dim=0)
            # [B, 1, 1, V] -> [B, V]
            log_prob = (self.joint(expand_tn_out, pn_out)
                        ).squeeze(1).squeeze(1)

            # [B,]
            distr_blank = log_prob[:, self.blank_id]
            for i, _log_p in enumerate(distr_blank):
                cur_hypo = sliced_B[original_indices[i]].clone()
                cur_hypo.score += _log_p
                buffer.update(cur_hypo)
                if group_t[map_rel_index_to_groupid[i]] == T-1:
                    F.append(cur_hypo)

            if use_lm:
                lm_out = torch.cat(group_lm_out, dim=0)
                # [B, 1, V] -> [B, V]
                log_prob = log_prob + lm_out.squeeze(1) * self.lm_weight

            # mask the blank id postion, so that we won't select it in top-k hypos
            log_prob[:, self.blank_id].fill_(float('-inf'))
            V = log_prob.size(1)    # type: int
            # add the hypo score here, then we can get the topk over all beams
            # [B, V] + [B, 1] -> [B, V]
            log_prob += collect_scores([sliced_B[i]
                                       for i in original_indices], dummy_tensor).unsqueeze(1)
            logp_targets, unnormal_tokens = torch.topk(
                log_prob.view(-1), k=self.beam_size, dim=-1)

            rel_indices = torch.div(
                unnormal_tokens, V, rounding_mode='floor').tolist()    # batch id
            tokens = torch.remainder(unnormal_tokens, V)
            for i, _log_p, tok in zip(rel_indices, logp_targets, tokens):
                # this judge can be remove
                if tok == self.blank_id:
                    continue
                cur_hypo = sliced_B[original_indices[i]].clone()
                # NOTE: assign, not add
                cur_hypo.score = _log_p
                cur_hypo.cache['pn_state'] = self.decoder.get_state_from_batch(
                    group_pn_state[map_rel_index_to_groupid[i]], map_rel_index_to_batchid[i])
                t_cache = {
                    'pn_out': pn_out[i:i+1],
                    'pn_state': cur_hypo.cache['pn_state']
                }
                if use_lm:
                    cur_hypo.cache['lm_state'] = self.lm.get_state_from_batch(
                        group_lm_state[map_rel_index_to_groupid[i]], map_rel_index_to_batchid[i])
                    t_cache.update({
                        'lm_out': lm_out[i:i+1],
                        'lm_state': cur_hypo.cache['lm_state']
                    })
                prefix_cache.update(cur_hypo.pred, t_cache)
                cur_hypo.pred.append(tok.item())
                buffer.update(cur_hypo)

            B = recombine_hypo(buffer.getBeam())
            prefix_cache.pruneShorterThan(min([len(hypo) for hypo in B]))

        if F == []:
            F = B
        Nbest = sorted(F, key=lambda item: item.score,
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
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        hs : No limit
            The memory variables are generated in this timestep.
            (e.g., RNN hidden states).
        """

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
    if prefix_cache is not None:
        in_cache = []
        for id, hypo in hypos_with_index:
            trycache = prefix_cache.getCache(hypo.pred)
            if trycache is None:
                continue
            in_cache.append((id, hypo))
        for id, _ in in_cache[::-1]:
            hypos_with_index.pop(id)

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
            cached_out.append((_index, _batched_out, _batched_states))

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
