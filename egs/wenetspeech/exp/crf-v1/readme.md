### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 83.45
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* data prepare and decode: `/mnt/workspace/zhenghh/CAT/egs/wenetspeech/run.sh`
* The dev set was processed in error, so the monitor plotting of dev loss is meaningless. However, this didn't affect the result, since I take the last 10 checkpoints for model averaging.

### Result
```
1-gram word-level LM
%WER 9.78 [ 10241 / 104765, 188 ins, 501 del, 9552 sub ] exp/crf-v1/decode/decode_aishell-test/cer_11_1.0
%WER 22.35 [ 49219 / 220228, 1921 ins, 18393 del, 28905 sub ] exp/crf-v1/decode/decode_test_meeting/cer_7_0.0
%WER 16.55 [ 68465 / 413621, 2962 ins, 13718 del, 51785 sub ] exp/crf-v1/decode/decode_test_net/cer_9_0.0
%WER 13.55 [ 44202 / 326330, 3990 ins, 12273 del, 27939 sub ] exp/crf-v1/decode/decode_dev/cer_6_0.0

3-gram word-level LM
%WER 6.83 [ 7156 / 104765, 192 ins, 426 del, 6538 sub ] exp/crf-v1/decode/decode_aishell-test/cer_12_1.0
%WER 11.38 [ 37139 / 326330, 3957 ins, 12254 del, 20928 sub ] exp/crf-v1/decode/decode_dev/cer_7_0.0
%WER 20.54 [ 45227 / 220228, 2058 ins, 17963 del, 25206 sub ] exp/crf-v1/decode/decode_test_meeting/cer_7_0.0                                                                           
%WER 13.49 [ 55812 / 413621, 3259 ins, 12231 del, 40322 sub ] exp/crf-v1/decode/decode_test_net/cer_9_0.0:q
```

### Monitor figure
![monitor](./monitor.png)
