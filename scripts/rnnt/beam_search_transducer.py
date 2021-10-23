"""Decoders and output normalization for Transducer sequence.

Author:
    Abdelwahab HEBA 2020
    Sung-Lin Yeh 2020

From speechbrain
"""
import copy
import yaml
from typing import Dict, Union, List, Optional
from transducer_train import JointNet

import torch
from typing import Literal


class Hypothesis():
    def __init__(self, pred: List[int], score: Union[torch.Tensor, float], cache: Union[dict, None]) -> None:
        self.pred = pred
        self.score = score
        self.cache = cache
        pass

    def clone(self):
        new_hypo = Hypothesis(
            self.pred[:],
            self.score if isinstance(
                self.score, float) else self.score.clone(),
            self.cache.copy()
        )
        return new_hypo


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
            print("Prefix:", pref)
            print(self)
            raise RuntimeError("Trying to update existing state.")
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

    def __str__(self) -> str:
        cache = {}
        for k in self._cache.keys():
            cache[k] = 'cache'

        return yaml.dump(cache, default_flow_style=False)


class TransducerBeamSearcher(torch.nn.Module):

    def __init__(
        self,
        decoder,
        joint: JointNet,
        blank_id: int = 0,
        bos_id: int = 0,
        beam_size: int = 5,
        nbest: int = 1,
        algo: Literal['default', 'lc'] = 'default',
        prefix_merge: bool = True,
        lm_module: Optional[torch.nn.Module] = None,
        lm_weight: float = 0.0,
        state_beam: float = 2.3,
        expand_beam: float = 2.3,
        temperature: float = 1.0
    ):
        super(TransducerBeamSearcher, self).__init__()
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

        self.state_beam = state_beam
        self.expand_beam = expand_beam

        self.is_latency_control = False
        self.is_prefix = prefix_merge
        if algo == 'default':
            self.searcher = self.transducer_beam_search_decode
        elif algo == 'lc':
            # latency controlled beam search
            self.is_latency_control = True
            self.searcher = self.transducer_beam_search_decode
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

    def transducer_beam_search_decode(self, tn_output: torch.Tensor):
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
                        process_hyps, key=lambda x: x.score / len(x.pred))

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if self.is_latency_control and len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps, key=lambda x: x.score / len(x.pred))
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
                    logp_targets, positions = torch.topk(
                        log_probs.view(-1), k=self.beam_size, dim=-1
                    )

                    if self.is_latency_control:
                        best_logp = (
                            logp_targets[0]
                            if positions[0] != blank
                            else logp_targets[1]
                        )

                    # Extend hyp by selection
                    for log_p, tok in zip(logp_targets, positions):
                        topk_hyp = a_best_hyp.clone()  # type: Hypothesis
                        topk_hyp.score += log_p

                        if tok == self.blank_id:
                            # prune the beam with same prefix
                            if self.is_prefix:
                                beam = None
                                for _beam in beam_hyps:
                                    if _beam.pred == topk_hyp.pred:
                                        beam = _beam
                                        break
                                if beam is None:
                                    beam_hyps.append(topk_hyp)
                                else:
                                    beam.score = torch.logaddexp(
                                        beam.score, topk_hyp.score)
                            else:
                                beam_hyps.append(topk_hyp)
                            continue

                        if (not self.is_latency_control) or (self.is_latency_control and log_p >= best_logp - self.expand_beam):
                            topk_hyp.pred.append(tok.item())
                            topk_hyp.cache["dec"] = hidden
                            if self.lm_weight > 0:
                                topk_hyp.cache["lm"] = hidden_lm

                                topk_hyp.score += (
                                    self.lm_weight
                                    * log_probs_lm[0, 0, tok])

                            process_hyps.append(topk_hyp)

            del prefix_cache
            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x.score/len(x.pred),
                reverse=True,
            )[: self.nbest]
            nbest_batch.append([hyp.pred[1:] for hyp in nbest_hyps])
            nbest_batch_score.append([hyp.score/len(hyp.pred)
                                     for hyp in nbest_hyps])

        return (nbest_batch, nbest_batch_score)

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
