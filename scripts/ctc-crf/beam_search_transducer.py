"""Decoders and output normalization for Transducer sequence.

Author:
    Abdelwahab HEBA 2020
    Sung-Lin Yeh 2020

From speechbrain
"""
import torch


class TransducerBeamSearcher(torch.nn.Module):

    def __init__(
        self,
        transducer,
        blank_id,
        beam_size=5,
        nbest=1,
        lm_module=None,
        lm_weight=0.0,
        state_beam=2.3,
        expand_beam=2.3,
    ):
        super(TransducerBeamSearcher, self).__init__()
        self.transducer = transducer
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.nbest = nbest
        self.lm = lm_module
        self.lm_weight = lm_weight

        if lm_module is None and lm_weight > 0.0:
            raise ValueError("Language model is not provided.")

        self.state_beam = state_beam
        self.expand_beam = expand_beam

        self.searcher = self.transducer_beam_search_decode

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

    def transducer_beam_search_decode(self, tn_output):
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

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """

        # min between beam and max_target_lent
        nbest_batch = []
        nbest_batch_score = []
        for i_batch in range(tn_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            blank = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            input_PN = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            # First forward-pass on PN
            hyp = {
                "prediction": [self.blank_id],
                "logp_score": 0.0,
                "hidden_dec": None,
            }
            if self.lm_weight > 0:
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)
            beam_hyps = [hyp]

            # For each time step
            for t_step in range(tn_output.size(1)):
                # get hyps for extension
                process_hyps = beam_hyps
                beam_hyps = []
                while True:
                    if len(beam_hyps) >= self.beam_size:
                        break
                    # Add norm score
                    a_best_hyp = max(
                        process_hyps,
                        key=lambda x: x["logp_score"] / len(x["prediction"]),
                    )

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps,
                            key=lambda x: x["logp_score"]
                            / len(x["prediction"]),
                        )
                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    # remove best hyp from process_hyps
                    process_hyps.remove(a_best_hyp)

                    # forward PN
                    input_PN[0, 0] = a_best_hyp["prediction"][-1]
                    out_PN, hidden = self._forward_PN(
                        input_PN,
                        a_best_hyp["hidden_dec"],
                    )
                    log_probs = self._joint_forward_step(
                        tn_output[i_batch, t_step, :], out_PN)

                    if self.lm_weight > 0:
                        log_probs_lm, hidden_lm = self._lm_forward_step(
                            input_PN, a_best_hyp["hidden_lm"]
                        )

                    # Sort outputs at time
                    logp_targets, positions = torch.topk(
                        log_probs.view(-1), k=self.beam_size, dim=-1
                    )
                    # logp_targets, positions = torch.topk(
                    #     log_probs.view(-1), k=log_probs.size(0), dim=-1
                    # )

                    best_logp = (
                        logp_targets[0]
                        if positions[0] != blank
                        else logp_targets[1]
                    )

                    # Extend hyp by selection
                    for j in range(logp_targets.size(0)):

                        # hyp
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"]
                            + logp_targets[j],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                        }

                        if positions[j] == self.blank_id:
                            beam_hyps.append(topk_hyp)
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
                            continue

                        if logp_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(positions[j].item())
                            topk_hyp["hidden_dec"] = hidden
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = hidden_lm
                                topk_hyp["logp_score"] += (
                                    self.lm_weight
                                    * log_probs_lm[0, 0, positions[j]]
                                )
                            process_hyps.append(topk_hyp)
            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[: self.nbest]
            all_predictions = []
            all_scores = []
            for hyp in nbest_hyps:
                all_predictions.append(hyp["prediction"][1:])
                all_scores.append(hyp["logp_score"] / len(hyp["prediction"]))
            nbest_batch.append(all_predictions)
            nbest_batch_score.append(all_scores)
        return (
            [nbest_utt[0] for nbest_utt in nbest_batch],
            torch.Tensor(
                [nbest_utt_score[0] for nbest_utt_score in nbest_batch_score]
            )
            .exp()
            .mean(),
            nbest_batch,
            nbest_batch_score,
        )

    def _joint_forward_step(self, out_TN, out_PN):
        """Join predictions (TN & PN)."""

        return self.transducer.joint(out_TN.view(-1), out_PN.view(-1))

    def _lm_forward_step(self, inp_tokens, memory):
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

        logits, hs = self.lm(inp_tokens, memory)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, hs

    def _forward_PN(self, input_PN, hidden=None):

        return self.transducer.decoder(input_PN, hidden)
