# Librispeech

[Previous results](./RESULT.md)

## Transducer

* Pre-processing: use Kaldi, the same as that of CTC-CRF training.
* SentencePiece tool to tokenize text.
* Modeling units: 1024 if no statement.
* ID: `<Prediction network> - <Transcription network> - <Joint network> - <id>`; CFM: Conformer.

| ID                                                         | test-clean | test-other | Ext LM   | Notes                                                        |
| ---------------------------------------------------------- | ---------- | ---------- | -------- | ------------------------------------------------------------ |
| [LSTM-CFM-FF-v1](exp/rnnt-v1-wp1024)                       | 5.63       | 13.33      | ---      | Baseline, train from scratch                                 |
| ↑                                                          | 5.44       | 12.78      | 0.1 LSTM | ↑                                                            |
| [LSTM-CFM-FF-v2](exp/rnnt-v4-joint-pretrained)             | 5.35       | 12.66      | ---      | Joint-training with pre-trained PN and TN                    |
| LSTM-CFM-FF-v3                                             | 11.66      | 21.64      | ---      | Pre-train and fix PN and TN                                  |
| [LSTM-CFM+LSTM-FF-v4](exp/rnnt-v5-lstmonAM)                | 6.44       | 15.06      | ---      | `v3` + stack 2 LSTM on TN                                    |
| [LSTM+LSTM-CFM+LSTM-FF-v5](exp/rnnt-v7-lstmOnBothSide/)    | 6.54       | 15.42      | ---      | `v3` + stack 2 LSTM on TN and PN respectively                |
| [EBD-CFM-FF-v6](exp/rnnt-v9-embeddingPN/)                  | 7.94       | 17.00      | ---      | Embedding PN. Train from scratch.                            |
| ↑                                                          | 6.28       | 14.29      | 0.3 LSTM | ↑                                                            |
| [LSTM-CFM-FF-v7](exp/rnnt-v11-longrun)                     | 5.55       | 12.47      | ---      | Google settings of Conformer-S. 40k steps.                   |
| ↑                                                          | 5.34       | 12.10      | 0.1 LSTM | ↑                                                            |
| Ref: Google CFM                                            | 2.7        | 6.3        | ---      | [ Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) |
| Ref: Efficient CFM                                         | 3.31       | 8.34       | ---      | [EfficientConformer: Efficient Conformer: Progressive Downsampling and Grouped Attention for Automatic Speech Recognition](https://github.com/burchim/EfficientConformer) |
| [LSTM-CFM-FF-v8](exp/rnnt-v14-noise-time-reduction-syncBN) | 5.08       | 10.96      | 0.1 LSTM | Sync BN. Time reduction. Variational noise. New SA Parameters. |
| [LSTM-CFM-FF-v9](exp/rnnt-v15-all-train)                   | 5.51       | 12.12      | 0.0 LSTM | `v8` + combine `tr95` and `cv05` as training set. Take dev-clean/dev-other as dev set. Smaller batch size 572->80. Disable time warp. Disable time reduction. Add weight decay in Adam. `0.0` LM weight is via grid search on dev set. |

