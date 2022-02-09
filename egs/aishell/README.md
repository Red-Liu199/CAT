## Data
170 hour普通话数据集，阅读类型

- [Github page](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell)
- 基础设置参考[[Librispeech]]
- 特征提取增加了3倍变速

## Result

format in `%WER (%WER with LM) [%WER oracle]`

| ID                                                                                                                                                | Notes                                                                          | test %CER         | averaging | \#params (M) |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------- | --------- | ------------ |
| [v1](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v1)                                                               | encoder: Conformer-S                                                           | 7.73\[5.89\]      | -         | 11.92        |
| [v2](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v2)                                                               | `v1` + peak factor\: 2.0 -> 1.0, warmup steps\: 10k -> 5k                      | 6.58              | best 10   | 11.92        |
| [v3](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v3)                                                               | encoder: refers to ESPNET                                                      | 5.32\[4.02\]      | best 10   | 89.58        |
| [v5](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v5)                                                               | `v3` + units\: BPE3500 -> char(~BPE4230), gradient clipping, attention dropout | 6.13\[4.01\]      | best 10   | 90.33        |
| [v7](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v7)                                                               | `v5` + rm gradient clipping, attention dropout\: 0.3 -> 0.2                    | 5.67\[3.80\]      | best 10   | ↑            |
| [v8](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v8)                                                               | `v7` + attention dropout\: 0.2 -> 0.1                                          | 5.46\[3.81\]      | last 10   | ↑            |
| [v10](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v10)                                                             | `v5` + rm spaces in transcript                                                 | 5.23\[4.00\]      | best 10   | ↑            |
| [v11](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v11)                                                             | `v10` + disable time warp, stop epoch\: 100 -> 80                              | 5.19\[3.99\]      | best 10   | ↑            |
| [v12](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v12)                                                             | `v10` + PN mask, stop epoch\: 100 -> 80                                        | 4.86\[3.46\]      | last 10   | ↑            |
| [v13](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v13)                                                             | `v12` + PN mask ratio: 0.2 -> 0.3, No. PN mask: 4 -> 3, stop epochs: 80 -> 100 | 4.81\[3.53\]      | best 10   | ↑            |
| [v14](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v14)                                                             | `v12` + subsample: conv2d -> vgg2l, stop epochs: 80 -> 100                     | **4.77** \[3.41\] | ↑         | 84.30        |
| [v15](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/rnnt-v15)                                                             | `v14` + add `<bos>` token + 5-gram LM trained on transcript                    | 4.82/4.69         | ↑         | ↑            |
| [CTC-v1](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/ctc-v1)                                                            | use the same encoder and tokenizer as `v15`                                    | 5.75/5.38         | ↑         | 68.19        |
| [ESPnet-1](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md#conformer-transducer-with-auxiliary-task-ctc-weight--05)      | RNN-T + CTC                                                                    | 4.8               | ?         | ~84.30       |
| [ESPnet-2](https://github.com/espnet/espnet/tree/master/egs2/aishell/asr1#conformer--specaug--speed-perturbation-featsraw-n_fft512-hop_length128) | CTC/Attention, result with transformer LM in `()`                              | 4.9(4.7)          | ?         | ?            |

- 特别要注意的是，使用BPE3500建模时，设置的SentencePiece覆盖率为0.9995，使用char建模时覆盖率设置为1
- `v1`和`v2`是初步的实验，主要用于验证代码，数据处理和模型平均等；
- `v3`部分参考了ESPNET（模型、scheduler、SpecAug参数等），~~仍是当前最好的结果~~
- `v5`相对`v3`增加了若干措施，改变了建模单元，性能出现显著恶化，根据后续实验分析，其中梯度裁剪影响较小
- `v5` -> `v7` -> `v8`: attention dropout似乎没有好处，奇怪的是，char建模的WER很好，但CER差
- `v10`去除空格之后，结果有一定的提升，但是仍然达不到ESPNET的5.0，且出现明显过拟合
- `v11`是关于time warp的消融实验，结果上看差距不大，可能也是Google不使用Time warp的原因
- `v12`基于`v10`过拟合的观察，在PN到Joint network之间增加了类似SpecAug的mask，其参数是经验选取的，没有微调，CER相对下降明显（~7% rel，5.21->4.85）。比较意思的是，增加了PN mask之后，在训练初始阶段`v12`模型dev loss反而更好，这一点是反直觉的：
   ![[compare-v10-v10+PN_mask.png]]
   结合此前一些其他的工作，我认为可能由于当前PN和TN均采用了同一学习率和优化器，而PN往往使用较浅的网络，使得PN模型的参数更新相对TN而言方差较大，以至于PN参数更新初始不稳定，增加PN mask之后，可以认为PN参数更新时梯度变小了，其参数更新的方差也就相应减小了。如果这一思考是正确的，那么理论上给PN使用更小的学习率也可以达到类似的效果。
   
## LM fusion
- Extra corpus: [THUCNEWS](http://thuctc.thunlp.org/#%E8%8E%B7%E5%8F%96%E9%93%BE%E6%8E%A5)
	- ~20M lines (include empty lines) in total.
	- unclean corpus, including Chinese/English punctuations, numbers (will be cast into `<unk>` using AISHELL-1 tokenizer).

- Processing of `<unk>`	: replaced to white space. e.g.
	- `你好，世界` -> `你好 世界`

- Fusion formula, fix $\beta=0.6$ in experiments:
$$
\begin{aligned}
&\text{without LM: } \hat{y} = \mathop{\arg\max}_y (\log P_{RNN-T}(Y|X)) \\
&\text{with LM: } \hat{y} = \mathop{\arg\max}_y (\log P_{RNN-T}(Y|X) + \lambda \log P_{LM}(Y) + \beta |Y|)
\end{aligned}
$$

- LM:
	- [lm-v4](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/lm-v4): trained on extra corpus, 5-gram, 11GB on disk.
	- [lm-v5](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/aishell/exp/lm-v5): trained on aishell-1 training set, 5-gram, 73MB on disk.

| Model | $\lambda$ | ppl   | CER (%) | RTF  |
| ----- | --------- | ----- | ------- | ---- |
| no lm | -         | -     | 4.82    | 0.53 |
| lm-v5 | 0.15      | 77.22 | 4.69    | 1.11 |
| lm-v4 | 0.50      | 64.97 | 3.67    | 1.30 |
