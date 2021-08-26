# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from typing import Union, Tuple, Sequence


class OpenspeechBeamSearchBase(nn.Module):
    """
    Openspeech's beam-search base class. Implement the methods required for beamsearch.
    You have to implement `forward` method.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, beam_size: int, sos_id: int = 0, pad_id: int = -1, eos_id: int = -2):
        super(OpenspeechBeamSearchBase, self).__init__()

        self.beam_size = beam_size
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.ongoing_beams = None
        self.cumulative_ps = None

    def _inflate(self, tensor: torch.Tensor, n_repeat: int, dim: int) -> torch.Tensor:
        repeat_dims = [1] * len(tensor.size())
        repeat_dims[dim] *= n_repeat
        return tensor.repeat(*repeat_dims)

    def _get_successor(
            self,
            current_ps: torch.Tensor,
            current_vs: torch.Tensor,
            finished_ids: tuple,
            num_successor: int,
            eos_count: int,
            k: int
    ) -> int:
        finished_batch_idx, finished_idx = finished_ids

        successor_ids = current_ps.topk(k + num_successor)[1]
        successor_idx = successor_ids[finished_batch_idx, -1]

        successor_p = current_ps[finished_batch_idx, successor_idx]
        successor_v = current_vs[finished_batch_idx, successor_idx]

        prev_status_idx = (successor_idx // k)
        prev_status = self.ongoing_beams[finished_batch_idx, prev_status_idx]
        prev_status = prev_status.view(-1)[:-1]

        successor = torch.cat([prev_status, successor_v.view(1)])

        if int(successor_v) == self.eos_id:
            self.finished[finished_batch_idx].append(successor)
            self.finished_ps[finished_batch_idx].append(successor_p)
            eos_count = self._get_successor(
                current_ps=current_ps,
                current_vs=current_vs,
                finished_ids=finished_ids,
                num_successor=num_successor + eos_count,
                eos_count=eos_count + 1,
                k=k,
            )

        else:
            self.ongoing_beams[finished_batch_idx, finished_idx] = successor
            self.cumulative_ps[finished_batch_idx, finished_idx] = successor_p

        return eos_count

    def _get_hypothesis(self):
        predictions = list()

        for batch_idx, batch in enumerate(self.finished):
            # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
            if len(batch) == 0:
                prob_batch = self.cumulative_ps[batch_idx]
                top_beam_idx = int(prob_batch.topk(1)[1])
                predictions.append(self.ongoing_beams[batch_idx, top_beam_idx])

            # bring highest probability sentence
            else:
                top_beam_idx = int(torch.FloatTensor(
                    self.finished_ps[batch_idx]).topk(1)[1])
                predictions.append(self.finished[batch_idx][top_beam_idx])

        predictions = self._fill_sequence(predictions)
        return predictions

    def _is_all_finished(self, k: int) -> bool:
        for done in self.finished:
            if len(done) < k:
                return False

        return True

    def _fill_sequence(self, y_hats: list) -> torch.Tensor:
        batch_size = len(y_hats)
        max_length = -1

        for y_hat in y_hats:
            if len(y_hat) > max_length:
                max_length = len(y_hat)

        matched = torch.zeros((batch_size, max_length), dtype=torch.long)

        for batch_idx, y_hat in enumerate(y_hats):
            matched[batch_idx, :len(y_hat)] = y_hat
            matched[batch_idx, len(y_hat):] = int(self.pad_id)

        return matched

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BeamSearchRNNTransducer(OpenspeechBeamSearchBase):
    r"""
    RNN Transducer Beam Search
    Reference: RNN-T FOR LATENCY CONTROLLED ASR WITH IMPROVED BEAM SEARCH (https://arxiv.org/pdf/1911.01629.pdf)

    Args: joint, decoder, beam_size, expand_beam, state_beam, blank_id
        joint: joint `encoder_outputs` and `decoder_outputs`
        decoder (TransformerTransducerDecoder): base decoder of transformer transducer model.
        beam_size (int): size of beam.
        expand_beam (int): The threshold coefficient to limit the number of expanded hypotheses.
        state_beam (int): The threshold coefficient to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (ongoing_beams)
        blank_id (int): blank id

    Inputs: encoder_output, max_length
        encoder_output (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(seq_length, dimension)``
        max_length (int): max decoding time step

    Returns:
        * predictions (torch.LongTensor): model predictions.
    """

    def __init__(
            self,
            transducer,
            beam_size: int = 3,
            expand_beam: float = 2.3,
            state_beam: float = 2.3,
            blank_id: int = 0,
    ) -> None:
        super(BeamSearchRNNTransducer, self).__init__(beam_size)
        self._trans = transducer
        self.expand_beam = expand_beam
        self.state_beam = state_beam
        self.blank_id = blank_id

    def forward(self, encoder_outputs: torch.Tensor, max_length: int):
        r"""
        Beam search decoding.

        Inputs: encoder_output, max_length
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predictions (torch.LongTensor): model predictions.
        """
        hypothesis = list()
        hypothesis_score = list()

        for batch_idx in range(encoder_outputs.size(0)):
            blank = (
                torch.ones((1, 1), device=encoder_outputs.device,
                           dtype=torch.long) * self.blank_id
            )
            step_input = (
                torch.ones((1, 1), device=encoder_outputs.device,
                           dtype=torch.long) * self.sos_id
            )
            hyp = {
                "prediction": [self.sos_id],
                "logp_score": 0.0,
                "hidden_states": None,
            }
            ongoing_beams = [hyp]

            for t_step in range(max_length):
                process_hyps = ongoing_beams
                ongoing_beams = list()

                while True:
                    if len(ongoing_beams) >= self.beam_size:
                        break

                    a_best_hyp = max(
                        process_hyps, key=lambda x: x["logp_score"] / len(x["prediction"]))

                    if len(ongoing_beams) > 0:
                        b_best_hyp = max(
                            ongoing_beams,
                            key=lambda x: x["logp_score"] /
                            len(x["prediction"]),
                        )

                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]

                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    process_hyps.remove(a_best_hyp)

                    step_input[0, 0] = a_best_hyp["prediction"][-1]

                    step_outputs, hidden_states = self._trans.decoder(
                        step_input, a_best_hyp["hidden_states"])
                    log_probs = self._trans.joint(
                        encoder_outputs[batch_idx, t_step, :], step_outputs.view(-1))

                    topk_targets, topk_idx = log_probs.topk(k=self.beam_size)

                    if topk_idx[0] != blank:
                        best_logp = topk_targets[0]
                    else:
                        best_logp = topk_targets[1]

                    for j in range(topk_targets.size(0)):
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"] + topk_targets[j],
                            "hidden_states": a_best_hyp["hidden_states"],
                        }

                        if topk_idx[j] == self.blank_id:
                            ongoing_beams.append(topk_hyp)
                            continue

                        if topk_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(topk_idx[j].item())
                            topk_hyp["hidden_states"] = hidden_states
                            process_hyps.append(topk_hyp)

            ongoing_beams = sorted(
                ongoing_beams,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[0]

            hypothesis.append(torch.LongTensor(
                ongoing_beams["prediction"][1:]))
            hypothesis_score.append(
                ongoing_beams["logp_score"] / len(ongoing_beams["prediction"]))

        return self._fill_sequence(hypothesis)


class ConvMemBuffer(nn.Module):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], channels: int) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self._mem = torch.zeros((channels,)+kernel_size)
        self._last_t_u = [-1, -1]

    def append(self, t: int, u: int, state: torch.Tensor):
        if t < 0 or u < 0:
            raise RuntimeError(
                f"Invalid (t, u) is fed into buffer: ({t}, {u}).")

        if t == 0 and u == 0:
            self._last_t_u = [0, 0]
        elif t > self._last_t_u[0]:
            self._last_t_u[0] += 1
            self._mem.roll(-1, 1)
            self._mem[:, -1, :] = 0.
        elif u > self._last_t_u[1]:
            self._last_t_u[1] += 1
            self._mem.roll(-1, 2)
            self._mem[:, :, -1] = 0.
        else:
            raise RuntimeError(
                f"Illegal (t, u) is fed into buffer: ({t}, {u}).")

        self._mem[:, -1, -1] = state

    @property
    def mem(self) -> torch.Tensor:
        return self._mem

    def replica(self):
        newbuffer = ConvMemBuffer(self._mem.size()[1:], self._mem.size(0))
        newbuffer._mem = self._mem
        newbuffer._last_t_u = self._last_t_u[:]
        return newbuffer


class BeamSearchConvTransducer(OpenspeechBeamSearchBase):
    def __init__(
            self,
            transducer,
            kernel_size: Union[int, Tuple[int, int]],
            channels: int,
            beam_size: int = 3,
            expand_beam: float = 2.3,
            state_beam: float = 2.3,
            blank_id: int = 0,
    ) -> None:
        super(BeamSearchRNNTransducer, self).__init__(beam_size)
        self._trans = transducer
        self.expand_beam = expand_beam
        self.state_beam = state_beam
        self.blank_id = blank_id
        self._buffer = ConvMemBuffer(kernel_size, channels)

    def forward(self, encoder_outputs: torch.Tensor, max_length: int):
        r"""
        Beam search decoding.

        Inputs: encoder_output, max_length
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predictions (torch.LongTensor): model predictions.
        """
        hypothesis = list()
        hypothesis_score = list()

        for batch_idx in range(encoder_outputs.size(0)):
            blank = (
                torch.ones((1, 1), device=encoder_outputs.device,
                           dtype=torch.long) * self.blank_id
            )
            step_input = (
                torch.ones((1, 1), device=encoder_outputs.device,
                           dtype=torch.long) * self.sos_id
            )
            hyp = {
                "prediction": [self.sos_id],
                "logp_score": 0.0,
                "hidden_states": None,
                "mem_blocks": self._buffer.replica()
            }
            ongoing_beams = [hyp]

            for t_step in range(max_length):
                process_hyps = ongoing_beams
                ongoing_beams = list()

                while True:
                    if len(ongoing_beams) >= self.beam_size:
                        break

                    a_best_hyp = max(
                        process_hyps, key=lambda x: x["logp_score"] / len(x["prediction"]))

                    if len(ongoing_beams) > 0:
                        b_best_hyp = max(
                            ongoing_beams,
                            key=lambda x: x["logp_score"] /
                            len(x["prediction"]),
                        )

                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]

                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    process_hyps.remove(a_best_hyp)

                    step_input[0, 0] = a_best_hyp["prediction"][-1]

                    step_outputs, hidden_states = self._trans.decoder(
                        step_input, a_best_hyp["hidden_states"])
                    log_probs, mem_blocks = self._trans.joint(
                        encoder_outputs[batch_idx, t_step, :], step_outputs.view(-1), a_best_hyp["mem_blocks"])

                    topk_targets, topk_idx = log_probs.topk(k=self.beam_size)

                    if topk_idx[0] != blank:
                        best_logp = topk_targets[0]
                    else:
                        best_logp = topk_targets[1]

                    for j in range(topk_targets.size(0)):
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"] + topk_targets[j],
                            "hidden_states": a_best_hyp["hidden_states"],
                            "mem_blocks": a_best_hyp["mem_blocks"].replica()
                        }

                        if topk_idx[j] == self.blank_id:
                            ongoing_beams.append(topk_hyp)
                            continue

                        if topk_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(topk_idx[j].item())
                            topk_hyp["hidden_states"] = hidden_states
                            topk_hyp["mem_blocks"].append(t_step, len(
                                topk_hyp["prediction"])-1, mem_blocks)
                            process_hyps.append(topk_hyp)

            ongoing_beams = sorted(
                ongoing_beams,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[0]

            hypothesis.append(torch.LongTensor(
                ongoing_beams["prediction"][1:]))
            hypothesis_score.append(
                ongoing_beams["logp_score"] / len(ongoing_beams["prediction"]))

        return self._fill_sequence(hypothesis)
