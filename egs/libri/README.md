960 hour英文数据集，阅读类型

- [Github page](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri)
- 特征提取：80维FBank+CMVN
- PN: prediction network, TN: transcription network, or encoder
- If no further statement, joint network is feed-forward model, PN is LSTM and TN is Conformer, they just differ in hidden size/number of layers in experiments.


Results in `%WER (%WER with LM) / %WER on averaging model [%WER oracle]`

| ID                                                                                                 | Notes                                                                                                    | test-clean       | test-other       | ext LM         | \# params (M)           |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------- | ---------------- | -------------- | ----------------------- |
| [v1](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v1-wp1024)           | BPE1024, all from scratch                                                                                | 5.63             | 13.33            | -              | 13.11                   |
| v3                                                                                                 | `v1` + pretrained and fixed PN & TN                                                                      | 11.66            | 21.64            | -              | trainable\:? all\:13.11 |
| [v4](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v4-joint-pretrained) | `v1` + pretrained PN & TN                                                                                | 5.35             | 12.66            | -              | 13.11                   |
| [v5](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v5-lstmonAM)         | `v1` + pretrained and fixed PN & TN, 2 extra LSTM on TN                                                  | 6.44             | 15.06            | -              | trainable\:5.52         |
| [v7](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v7-lstmOnBothSide)   | `v5` + 2 extra LSTM on PN                                                                                | 6.54             | 15.42            | -              | trainable\:9.72         |
| [v9](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v9-embeddingPN)      | `v1` + replace PN with embedding only layer                                                              | 7.94(6.28)       | 17.00(14.29)     | 0.3 LSTM       | 8.65                    |
| [v11](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v11-longrun)        | BPE4096 + stop at 40k steps                                                                              | 5.55(5.34)       | 12.47(12.10)     | 0.1 LSTM       | 13.06                   |
| [v19](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v19)                | `v1` + rm redundant linear layers, variational noise, sync BN, time reduction 2, batch size\: 560 -> 512 | 4.64             | 11.54            | -              | 10.33                   |
| [v20](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v19)                | `v19` + peak factor\: 1.0 -> 5.0, stop lr\: 1e-6 -> 1e-3                                                 | 4.50/4.30        | 10.76/10.17      | -              | 10.33                   |
| [v21](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v21)                | `v20` + stop at 200 epochs                                                                               | 4.05/3.81        | 10.03/9.38       | -              | 10.33                   |
| [v23](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v23)                | `v21` + fix dataloader bug                                                                               | 3.75/3.65        | 9.50/9.01        | -              | 10.33                   |
| [v24](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v24)                | follow settings of aishell `v14`                                                                         | 3.19/2.97        | 7.89/7.31        | -              | 81.01                   |
| [v25](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v25)                | `v24` + adjust scheduler                                                                                 | 3.01/2.70        | 7.46/6.70        | -              | ↑                       |
| [v26](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v26)                | same encoder as the ASRU advancing libri model                                                           | 2.68\[1.99\]     | 6.28\[5.22\]     | -              | 62.24                   |
| [v27](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v27)                | 2048 batch size + large conformer                                                                        | **2.37\[2.00\]** | **5.46\[4.63\]** | -              | 120.42                  |
| [ESPNET-2](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1#with-transformer-lm) | Conformer encoder                                                                                        | 2.6(2.1)         | 6.0(4.7)         | transformer LM | ?                       |

- `v24` 训得不太好，学习率太小
- `v25` 仍有改进空间，～100epoch即过拟合

Beam search, test-clean, `# frames=1940626` -> `total time=19406.26 s`
- GPU: 8 $\times$ RTX 3090
- CPU: 48-core Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz

| Method | Prefix merge | size | WER  | Oracle | Time (s) | Hardware | RTF  |
| ------ | ------------ | ---- | ---- | ------ | -------- | -------- | ---- |
| Native | ✘            | 5    | 2.70 | 2.58   | 465.29   | **GPU**  | 0.19 |
| Native | ✔            | 5    | 2.71 | 2.00   | 576.06   | **GPU**  | 0.24 |
| Native | ✔            | 5    | 2.71 | 2.00   | 290.88   | CPU      | 0.72 |
| Native | ✔            | 10   | 2.70 | 1.47   | 2993.13  | **GPU**  | 1.23 |
| Native | ✔            | 10   | 2.70 | 1.47   | 490.09   | CPU      | 1.21 |
| LC     | ✘            | 5    | 2.70 | 2.60   | 275.84   | **GPU**  | 0.11 |
| LC     | ✔            | 5    | 2.69 | 2.36   | 159.03   | **GPU**  | 0.07 |
| LC     | ✔            | 5    | 2.69 | 2.36   | 215.09   | CPU      | 0.53 |
| LC     | ✔            | 10   | 2.70 | 2.20   | 253.27   | **GPU**  | 0.10 |
| LC     | ✔            | 10   | 2.70 | 2.20   | 223.56   | CPU      | 0.55 | 
| LC     | ✔            | 20   | 2.79 | 2.25   | 401.39   | **GPU**  | 0.17 |
