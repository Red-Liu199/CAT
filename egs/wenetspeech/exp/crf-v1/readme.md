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
%WER 11.15 [ 36536 / 327711, 2727 ins, 12405 del, 21404 sub ] exp/crf-v1/phn-tgprune/dev/cer_7_0.0
%WER 13.38 [ 55453 / 414392, 2590 ins, 12334 del, 40529 sub ] exp/crf-v1/phn-tgprune/test_net/cer_9_0.0
%WER 20.52 [ 45219 / 220338, 1968 ins, 17993 del, 25258 sub ] exp/crf-v1/phn-tgprune/test_meeting/cer_7_0.0
%WER 6.83 [ 7153 / 104765, 191 ins, 430 del, 6532 sub ] exp/crf-v1/phn-tgprune/aishell-test/cer_12_1.0

+aishell word
%WER 5.53 [ 5795 / 104765, 191 ins, 370 del, 5234 sub ] exp/crf-v1/phn-word-3gram/aishell-test/cer_12_0.5

+aishell char
%WER 6.86 [ 7192 / 104765, 160 ins, 432 del, 6600 sub ] exp/crf-v1/phn-char-5gram/aishell-test/cer_12_1.0

+5mil corpus
%WER 10.46 [ 34273 / 327711, 2744 ins, 12072 del, 19457 sub ] exp/crf-v1/phn-word-3gram/dev/cer_7_0.0
%WER 12.53 [ 51918 / 414392, 2512 ins, 12874 del, 36532 sub ] exp/crf-v1/phn-word-3gram/test_net/cer_10_0.0
%WER 19.82 [ 43681 / 220338, 1996 ins, 17618 del, 24067 sub ] exp/crf-v1/phn-word-3gram/test_meeting/cer_7_0.0
%WER 6.09 [ 6378 / 104765, 206 ins, 357 del, 5815 sub ] exp/crf-v1/phn-word-3gram/aishell-test/cer_11_1.0

+10mil corpus
%WER 10.46 [ 34284 / 327711, 2729 ins, 12149 del, 19406 sub ] exp/crf-v1/phn-word-3gram/dev/cer_7_0.0
%WER 12.28 [ 50880 / 414392, 2462 ins, 12819 del, 35599 sub ] exp/crf-v1/phn-word-3gram/test_net/cer_10_0.0
%WER 19.71 [ 43435 / 220338, 1973 ins, 17742 del, 23720 sub ] exp/crf-v1/phn-word-3gram/test_meeting/cer_7_0.0
%WER 5.99 [ 6276 / 104765, 194 ins, 371 del, 5711 sub ] exp/crf-v1/phn-word-3gram/aishell-test/cer_11_1.0

+15mil corpus
%WER 10.35 [ 33926 / 327711, 2750 ins, 12188 del, 18988 sub ] exp/crf-v1/phn-word-3gram/dev/cer_7_0.0
%WER 12.15 [ 50353 / 414392, 2480 ins, 12757 del, 35116 sub ] exp/crf-v1/phn-word-3gram/test_net/cer_10_0.0
%WER 19.56 [ 43104 / 220338, 1948 ins, 17778 del, 23378 sub ] exp/crf-v1/phn-word-3gram/test_meeting/cer_7_0.0
%WER 5.74 [ 6009 / 104765, 194 ins, 375 del, 5440 sub ] exp/crf-v1/phn-word-3gram/aishell-test/cer_11_1.0

char trans
%WER 12.14 [ 39792 / 327711, 2775 ins, 12515 del, 24502 sub ] exp/crf-v1/phn-char-5gram/dev/cer_7_0.0
%WER 14.44 [ 59848 / 414392, 2705 ins, 12336 del, 44807 sub ] exp/crf-v1/phn-char-5gram/test_net/cer_9_0.0
%WER 21.62 [ 47629 / 220338, 2037 ins, 18180 del, 27412 sub ] exp/crf-v1/phn-char-5gram/test_meeting/cer_7_0.0
%WER 8.52 [ 8929 / 104765, 164 ins, 541 del, 8224 sub ] exp/crf-v1/phn-char-5gram/aishell-test/cer_13_1.0
```

### Monitor figure
![monitor](./monitor.png)
