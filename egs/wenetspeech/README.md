## 数据
[wenet-e2e/WenetSpeech: A 10000+ hours dataset for Chinese speech recognition (github.com)](https://github.com/wenet-e2e/WenetSpeech)  
使用wenet speech train-m （～1000小时）作为训练集  
数据处理：

-   默认使用80-Fbank特征，数据集均为16kHz采样
-   使用CMVN，默认不采用3倍变速

## Perf. (PPL) of LMs
5-gram LMs trained on different datasets:

| training set                      | aishell-test | wenet-test | \# utt (M) |
| --------------------------------- | ------------ | ---------- | ---------- |
| aishell                           | 58.66        | 194.64     | ~0.12      |
| transcript (wenet-m)              | 98.96        | 70.67      | ~1.5       |
| wenet-m + 5 M lines from wenet-l  | 79.83        | 60.11      | ~6.5       |
| wenet-m + 10 M lines from wenet-l | 78.26        | 57.52      | ~11.5      |
| wenet-m + 15 M lines from wenet-l | 72.42        | 56.16      | ~16.5      | 

## Performance (%CER)  comparison
- NN model: same encoder
- Modeling units: 
	- CTC-char and RNN-T: ~5000 char;
	- CTC-phone and CRF: 162 phone.
- 1-pass decode: 
	- CTC-char and RNN-T: beam search
	- CTC-phone and CRF: WFST

| Topo      | LM          | dev   | test-net | test-meeting | aishell-test |
| --------- | ----------- | ----- | -------- | ------------ | ------------ |
| CTC-char  | -           | 12.24 | 15.44    | 24.04        | 8.78         |
| CTC-phone | 3-gram word | 13.38 | 16.96    | 26.43        | 8.18         |
| CTC-CRF   | 1-gram word | 13.55 | 16.55    | 22.35        | 9.78         |
| CTC-CRF   | 3-gram word | 11.38 | 13.49    | 20.54        | 6.83         |
| CTC-CRF   | 5-gram word | 11.38 | 13.49    | 20.51        | 6.84         |
| RNN-T     | -           | 11.16 | 12.76    | 20.97        | 7.22         |
| RNN-T     | 5-gram char | 9.80  | 12.11    | 19.40        | 6.23         | 

## Comparison of decoding methods & LM
Perfermance is evaluated on AISHELL-test set. See [[#LM adaptation]] for the meaning of *baseline* and *final*

| ID  | Model | Unit  | Method              | LM         | baseline | final    | % Rel |
| --- | ----- | ----- | ------------------- | ---------- | -------- | -------- | ----- |
| 1   | CTC   | phone | FST                 | 3gram word | 8.22     | 6.76     | 17.8  |
| 2   | ↑     | ↑     | ↑                   | 5gram char | 9.82     | 8.08     | 17.7  |
| 3   | ↑     | char  | beam search         | 5gram char | 7.19     | 6.05     | 15.9  |
| 4   | ↑     | ↑     | FST                 | 3gram word | 6.67     | 5.89     | 11.7  |
| 5   | ↑     | ↑     | ↑                   | 5gram char | 7.08     | 6.26     | 11.6  |
| 6   | CRF   | phone | ↑                   | 3gram word | 6.86     | **5.63** | 17.9  |
| 7   | ↑     | ↑     | ↑                   | 5gram char | 8.54     | 6.94     | 18.7  |
| 9   | RNNT  | char  | beam search         | 5gram char | **6.23** | 5.86     | 5.9   |
| 10  | RNNT  | char  | beam search+rescore | 3gram word |          |          |       |

Conclusion:
- word vs. char LM: 1&2, 4&5, 6&7. All word LMs perform better than char LMs. Dictionary info helps.
- phone vs. char AM: 1&4, 2&5. Char modeling unit performs better.
- Beam search vs. FST: 3&5. Theoretically they're of the same performance. But results show that beam search is somewhat better.

## LM adaptation

- [ ] On language model integration capabilities of the hybrid and the RNN-T architectures
- [ ] On language model integration capabilities of CTC-CRF and RNN-T architectures

A -> B: model trained on A dataset, then, evaluated on B test set as **baseline**, evaluated on B test set with LM trained on B transcript as **final**

Decode:
- CTC-char and RNN-T: beam search for baseline; fusion with LM for final
- CTC-phone and CTC-CRF: decode with 3-gram LM trained on A transcript as baseline; decode with 3-gram LM trained on B transcript as final. (WFST)

`Rel = (baseline - final) / baseline`

mandarin: wenet-m -> aishell

see [[#Comparison of decoding methods LM]]

english: libri -> TLv2

| Topo    | Baseline | Final | %Rel |
| ------- | -------- | ----- | ---- |
| CTC     |          |       |      |
| CTC-CRF |          |       |      |
| RNN-T   | 11.16    | 10.24 | 8.2  | 



## LM integration
Test CER of LM integration on wenetspeech test-meeting

| LM training set                 | ppl   | RNN-T | CTC-CRF |
| ------------------------------- | ----- | ----- | ------- |
| baseline w/o LM                 | -     | 20.97 | 20.54   | 
| transcript (wenet-m)            | 70.67 |       |         |
| wenet-m + 5 M lines of wenet-l  | 60.11 |       |         |
| wenet-m + 10 M lines of wenet-l | 57.52 |       |         |
| wenet-m + 15 M lines of wenet-l | 56.16 |       |         |
