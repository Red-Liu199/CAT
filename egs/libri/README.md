## Data
- 960 hour英文数据集，阅读类型
- [Github page](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri)
- 特征提取：80维FBank+CMVN
- Text corpus数据准备

	```bash
	# in egs/libri/
	bash local/prepare_extra_corpus.sh
	```
- `hyper-p.json`中配置LM训练数据集
	```json
	"data": {
	
		"train": ["data/librispeech.txt", "train_960"],
		
		"dev": ["dev_clean", "dev_other"],
		
		"test": ["test_clean", "test_other"],
		...
	},
	...
	```
	其中`train, dev, test`分别设置训练、开发、测试集数据，以训练集数据为例，其中包含`data/librispeech.txt` 和 `train_960`。`utils/*_process.py[sh]` 脚本对此的处理逻辑是：
	1. 首先尝试直接将其解析为文件，如果该文件存在，则直接选取；例如`data/librispeech.txt`如果存在，则可以直接读取；
	2. 若第1步未找到文件，则尝试在CAT对应egs目录下搜索；例如`train_960`在当前目录没有这一文件，则会尝试在`tools/CAT/egs/libri/data`下寻找`train_960`目录，并取`train_960/text`作为数据
	第1步解析到的文件会直接被作为LM训练语料；第2步解析的文件会将居首用` `, `\t` 分割的句子id去除，作为训练语料。该解析过程代码位于[priorResolvePath](https://github.com/maxwellzh/Transducer-dev/blob/586501dfd9b01eb8300085812e77be846b4617b7/cat/utils/asr_process.py#L396-L428)
- 对ASR训练而言，声学特征和文本特征必须匹配，因此只支持上述第二种数据解析方式

## Result

format in `%WER (%WER with LM) / %WER on averaging model [%WER oracle]`

| ID                                                                                                 | Notes                                   | test-clean       | test-other       | ext LM         | \# params (M) |
| -------------------------------------------------------------------------------------------------- | --------------------------------------- | ---------------- | ---------------- | -------------- | ------------- |
| CRF-v1                                                                                             | CTC-CRF phone modeling, conformer model | 2.76             | 5.53             | -              | 115.12        |
| [v27](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/rnnt-v27)                | 2048 batch size + large conformer       | **2.37\[2.05\]** | **5.46\[4.55\]** | transformer LM | 120.42        |
| [ESPNET-2](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1#with-transformer-lm) | Conformer encoder                       | 2.6(2.1)         | 6.0(4.7)         | transformer LM | ?             |

## Beam search
test-clean, `# frames=1940626` -> `total time=19406.26 s`
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


## Decode and LM fusion

- Time is measured on server15 with 80 processes

LM integration formula:
$$
\begin{aligned}
&\text{without LM: } \hat{y} = \mathop{\arg\max}_y (\log P_{RNN-T}(Y|X)) \\
&\text{with LM: } \hat{y} = \mathop{\arg\max}_y (\log P_{RNN-T}(Y|X) + \alpha \log P_{LM}(Y) + \beta |Y|)
\end{aligned}
$$

Language model (char modeling if no further statement)

| ID                                                                                               | Model                  | Data                                | \#param (M)   | ppl   | word level ppl |
| ------------------------------------------------------------------------------------------------ | ---------------------- | ----------------------------------- | ------------- | ----- | -------------- |
| [lm-v10](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/lm-v10)             | Transformer            | Libri transcript + text book corpus | 87.42         | 13.25 | ?              |
| [lm-v11](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/lm-v11-trans-5gram) | 5-gram                 | Libri transcript                    | 489MB on disk | 32.20 | ?              |
| [lm-v12](https://github.com/maxwellzh/Transducer-dev/tree/main/egs/libri/exp/lm-v12-large-5ram)  | 5-gram                 | Libri transcript + text book corpus | 15GB on disk  | 23.15 | ?              | 
| -                                                                                                | char 5gram TRF         |                                     |               |       |                |
| -                                                                                                | word 3gram -> char TRF |                                     |               |       |                |

LM integration results (baseline result from `rnnt-v27`), $\alpha$ and $\beta$ are searched on `dev-other` set

| Model    | method  | $\alpha$ | $\beta$ | dev-clean | dev-other | test-clean | test-other |
| -------- | ------- | -------- | ------- | --------- | --------- | ---------- | ---------- |
| baseline | -       | -        | -       | 2.18      | 5.34      | 2.39       | 5.41       |
| lm-v10   | rescore | 0.75     | 0.88    | 1.90      | 4.34      | 2.05       | 4.55       |
| lm-v11   | rescore | 0.50     | 1.03    | 2.25      | 5.06      | 2.42       | 5.24       |
| lm-v12   | rescore | 0.63     | 1.19    | 2.10      | 4.76      | 2.20       | 4.96       | 
