### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 115.12
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* derived from `rnnt-v27`, trained with CTC-CRF phone-based

### Result
```
best 10
%WER 2.60 [ 1413 / 54402, 122 ins, 250 del, 1041 sub ] exp/crf-v1/decode_dev_clean_fglarge/wer_11_1.0
%WER 5.93 [ 3023 / 50948, 246 ins, 535 del, 2242 sub ] exp/crf-v1/decode_dev_other_fglarge/wer_17_0.0
%WER 2.93 [ 1539 / 52576, 132 ins, 299 del, 1108 sub ] exp/crf-v1/decode_test_clean_fglarge/wer_13_0.5
%WER 6.18 [ 3237 / 52343, 277 ins, 617 del, 2343 sub ] exp/crf-v1/decode_test_other_fglarge/wer_14_1.0

last 10
%WER 2.42 [ 1319 / 54402, 115 ins, 271 del, 933 sub ] exp/crf-v1//fglarge/dev_clean/wer_15_1.0
%WER 5.33 [ 2718 / 50948, 261 ins, 424 del, 2033 sub ] exp/crf-v1//fglarge/dev_other/wer_17_0.0
%WER 2.74 [ 1442 / 52576, 137 ins, 270 del, 1035 sub ] exp/crf-v1//fglarge/test_clean/wer_13_0.5
%WER 5.53 [ 2893 / 52343, 274 ins, 500 del, 2119 sub ] exp/crf-v1//fglarge/test_other/wer_14_0.5
%WER 12.24 [ 2176 / 17783, 529 ins, 431 del, 1216 sub ] exp/crf-v1//fglarge/tlv2-dev/wer_13_1.0
%WER 11.86 [ 3262 / 27500, 558 ins, 753 del, 1951 sub ] exp/crf-v1//fglarge/tlv2-test/wer_13_1.0

rescore with lm-v10
dev_clean   %SER 26.01 | %WER 2.06 [ 1120 / 54402, 143 ins, 82 del, 895 sub ]
dev_other   %SER 39.46 | %WER 4.52 [ 2303 / 50948, 277 ins, 181 del, 1845 sub ]
test_clean  %SER 27.56 | %WER 2.35 [ 1236 / 52576, 180 ins, 85 del, 971 sub ]
test_other  %SER 43.25 | %WER 4.84 [ 2532 / 52343, 301 ins, 187 del, 2044 sub ]
tlv2-dev    %SER 85.01 | %WER 11.86 [ 2109 / 17783, 626 ins, 265 del, 1218 sub ]
tlv2-test   %SER 75.15 | %WER 11.30 [ 3108 / 27500, 656 ins, 508 del, 1944 sub ]

tuned alpha/beta
%SER 25.82 | %WER 2.06 [ 1118 / 54402, 146 ins, 81 del, 891 sub ]     alpha = 1.625 | beta = -0.25
%SER 39.35 | %WER 4.51 [ 2297 / 50948, 267 ins, 186 del, 1844 sub ]   alpha = 2.5 | beta = -0.5
%SER 27.56 | %WER 2.35 [ 1236 / 52576, 180 ins, 84 del, 972 sub ]     alpha = 1.3125 | beta = -0.5
%SER 43.04 | %WER 4.80 [ 2511 / 52343, 296 ins, 195 del, 2020 sub ]   alpha = 1.6875 | beta = -0.5
%SER 84.62 | %WER 11.58 [ 2059 / 17783, 576 ins, 285 del, 1198 sub ]  alpha = 2.5 | beta = 0.25
%SER 75.06 | %WER 11.14 [ 3063 / 27500, 638 ins, 516 del, 1909 sub ]  alpha = 1.71875 | beta = 0.5

tuned with lm-v12
%SER 31.78 | %WER 2.54 [ 1383 / 54402, 148 ins, 227 del, 1008 sub ]   alpha = 1.0625 | beta = -0.5
%SER 45.64 | %WER 5.45 [ 2779 / 50948, 236 ins, 427 del, 2116 sub ]   alpha = 1.6875 | beta = 0.0
%SER 33.36 | %WER 2.81 [ 1476 / 52576, 147 ins, 250 del, 1079 sub ]   alpha = 1.03125 | beta = -0.75
%SER 48.89 | %WER 5.68 [ 2971 / 52343, 260 ins, 504 del, 2207 sub ]   alpha = 1.46875 | beta = -0.25
%SER 87.97 | %WER 12.27 [ 2182 / 17783, 526 ins, 405 del, 1251 sub ]  alpha = 1.25 | beta = -0.75
%SER 79.65 | %WER 12.21 [ 3357 / 27500, 602 ins, 676 del, 2079 sub ]  alpha = 1.375 | beta = 0.5


+tlv2 corpus
%WER 10.38 [ 1845 / 17783, 453 ins, 360 del, 1032 sub ] exp/crf-v1/tlv2-fg/tlv2-dev/wer_17_0.5
%WER 9.81 [ 2697 / 27500, 479 ins, 624 del, 1594 sub ] exp/crf-v1/tlv2-fg/tlv2-test/wer_16_0.5
```

### Monitor figure
![monitor](./monitor.png)
