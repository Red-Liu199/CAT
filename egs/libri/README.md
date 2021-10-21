960 hour英文数据集，阅读类型

- [Github page](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri)
- PN: prediction network, TN: transcription network, or encoder
- If no further statement, joint network is feed-forward model, PN is LSTM and TN is Conformer, they just differ in hidden size/number of layers in experiments.


Results in `%WER(%WER on averaging model)/%WER with fusion`

| ID                                                                                                 | Notes                                                                                                    | test-clean     | test-other     | ext LM   | \# params (M)           |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------- | -------------- | -------- | ----------------------- |
| [v1](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v1-wp1024)           | BPE1024, all from scratch                                                                                | 5.63           | 13.33          | -        | 13.11                   |
| v3                                                                                                 | `v1` + pretrained and fixed PN & TN                                                                      | 11.66          | 21.64          | -        | trainable\:? all\:13.11 |
| [v4](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v4-joint-pretrained) | `v1` + pretrained PN & TN                                                                                | 5.35           | 12.66          | -        | 13.11                   |
| [v5](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v5-lstmonAM)         | `v1` + pretrained and fixed PN & TN, 2 extra LSTM on TN                                                  | 6.44           | 15.06          | -        | trainable\:5.52         |
| [v7](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v7-lstmOnBothSide)   | `v5` + 2 extra LSTM on PN                                                                                | 6.54           | 15.42          | -        | trainable\:9.72         |
| [v9](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v9-embeddingPN)      | `v1` + replace PN with embedding only layer                                                              | 7.94/6.28      | 17.00/14.29    | 0.3 LSTM | 8.65                    |
| [v11](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v11-longrun)        | BPE4096 + stop at 40k steps                                                                              | 5.55/5.34      | 12.47/12.10    | 0.1 LSTM | 13.06                   |
| [v19](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v19)                | `v1` + rm redundant linear layers, variational noise, sync BN, time reduction 2, batch size\: 560 -> 512 | 4.64           | 11.54          | -        | 10.33                   |
| [v20](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v19)                | `v19` + peak factor\: 1.0 -> 5.0, stop lr\: 1e-6 -> 1e-3                                                 | 4.50(4.30)     | 10.76(10.17)   | -        | 10.33                   |
| [v21](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v21)                | `v20` + stop at 200 epochs                                                                               | 4.05(3.81)     | 10.03(9.38)    | -        | 10.33                   |
| [v23](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v23)                | `v21` + fix dataloader bug                                                                               | 3.75(3.65)     | 9.50(9.01)     | -        | 10.33                   |
| [v24](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v24)                | follow settings of aishell `v14`                                                                         | 3.19(2.97)     | 7.89(7.31)     | -        | 81.01                   |
| [v25](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v25)                | `v24` + adjust scheduler                                                                                 | **3.01(2.70)** | **7.46(6.70)** | -        | ↑                       |

- `v24` 训得不太好，学习率太小
- `v25` 仍有改进空间，～100epoch即过拟合