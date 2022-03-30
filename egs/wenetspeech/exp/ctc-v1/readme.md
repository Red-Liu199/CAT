### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 86.01
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* CTC topo of `rnnt-v1`

### Result
```
test_meeting    %SER 95.05 | %CER 24.04 [52979 / 220385, 1346 ins, 21329 del, 30304 sub ]
test_net        %SER 74.35 | %CER 15.44 [64172 / 415746, 2045 ins, 14216 del, 47911 sub ]
dev     %SER 76.34 | %CER 12.24 [40467 / 330498, 1272 ins, 15956 del, 23239 sub ]
aishell-test    %SER 59.98 | %CER 8.78 [9202 / 104765, 339 ins, 187 del, 8676 sub ]

+trans lm aishell-test
setting: alpha = 0.56 | beta = 0.88
%SER 49.60 | %CER 7.19 [7534 / 104765, 227 ins, 263 del, 7044 sub ]

```

### Tuning on aishell-1

```
alpha 0.4
aishell-test    %SER 44.41 | %CER 6.17 [6461 / 104765, 173 ins, 251 del, 6037 sub ]

alpha 0.5
aishell-test    %SER 43.41 | %CER 6.05 [6338 / 104765, 144 ins, 316 del, 5878 sub ]

alpha 0.55
aishell-test    %SER 43.14 | %CER 6.05 [6339 / 104765, 141 ins, 358 del, 5840 sub ]

alpha 0.6
aishell-test    %SER 43.17 | %CER 6.06 [6350 / 104765, 133 ins, 400 del, 5817 sub ]

```

LM adaptation

```
>>> no LM
    aishell-dev     %SER 55.20 | %CER 7.72 [ 15859 / 205341, 467 ins, 256 del, 15136 sub ]
    aishell-test    %SER 59.98 | %CER 8.78 [ 9201 / 104765, 338 ins, 187 del, 8676 sub ]

>>> SF  [0.5, 1.25]
    %SER 41.20 | %CER 5.61 [ 11517 / 205341, 349 ins, 309 del, 10859 sub ]
    %SER 43.80 | %CER 6.21 [ 6510 / 104765, 210 ins, 214 del, 6086 sub ]

>>> DR  [0.0, 0.5, 1.25]


>>> WLM

5-100   
    %SER 39.60 | %CER 5.26 [ 10808 / 205341, 326 ins, 269 del, 10213 sub ]  [-0.375, 0.625, 0.125]
    %SER 42.63 | %CER 5.96 [ 6243 / 104765, 167 ins, 236 del, 5840 sub ]

5-500
    %SER 40.07 | %CER 5.36 [ 11006 / 205341, 333 ins, 260 del, 10413 sub ]  [-0.375, 0.625, 0.0]
    %SER 42.91 | %CER 6.04 [ 6323 / 104765, 167 ins, 238 del, 5918 sub ]

5-1000
    %SER 40.23 | %CER 5.42 [ 11122 / 205341, 324 ins, 287 del, 10511 sub ]  [-0.375, 0.625, -0.25]
    %SER 43.00 | %CER 6.09 [ 6376 / 104765, 165 ins, 263 del, 5948 sub ]

5-1500
    %SER 40.46 | %CER 5.48 [ 11249 / 205341, 339 ins, 288 del, 10622 sub ]  [-0.375, 0.625, -0.25]
    %SER 43.10 | %CER 6.10 [ 6393 / 104765, 175 ins, 261 del, 5957 sub ]

5-2000
    %SER 40.88 | %CER 5.53 [ 11361 / 205341, 352 ins, 267 del, 10742 sub ]  [-0.125, 0.5, 0.75]
    %SER 43.56 | %CER 6.16 [ 6454 / 104765, 222 ins, 195 del, 6037 sub ]
```

LM integration

```
>>> no lm combine-set 9.13
    aishell-dev     %SER 55.20 | %CER 7.72 [ 15859 / 205341, 467 ins, 256 del, 15136 sub ]
    aishell-test    %SER 59.98 | %CER 8.78 [ 9201 / 104765, 338 ins, 187 del, 8676 sub ]
    speechio_asr_zh00000    %SER 76.79 | %CER 12.27 [ 2917 / 23765, 48 ins, 1826 del, 1043 sub ]
    speechio_asr_zh00001    %SER 56.86 | %CER 4.40 [ 6308 / 143203, 51 ins, 499 del, 5758 sub ]
    speechio_asr_zh00002    %SER 65.09 | %CER 9.77 [ 5038 / 51551, 256 ins, 1602 del, 3180 sub ]
    speechio_asr_zh00003    %SER 75.16 | %CER 8.72 [ 3233 / 37064, 41 ins, 97 del, 3095 sub ]
    speechio_asr_zh00004    %SER 67.73 | %CER 5.87 [ 2200 / 37506, 86 ins, 652 del, 1462 sub ]
    speechio_asr_zh00005    %SER 89.93 | %CER 10.49 [ 9792 / 93322, 220 ins, 4667 del, 4905 sub ]
    speechio_asr_zh00006    %SER 89.37 | %CER 22.32 [ 5875 / 26318, 261 ins, 1187 del, 4427 sub ]
    speechio_asr_zh00007    %SER 94.03 | %CER 23.45 [ 4417 / 18832, 133 ins, 1209 del, 3075 sub ]

>>> + LM-trans-5m
SF  %SER 54.43 | %CER 7.13 [ 38220 / 536326, 3153 ins, 8651 del, 26416 sub ]        [0.5, 4.5]
    %SER 40.71 | %CER 5.73 [ 6002 / 104765, 674 ins, 116 del, 5212 sub ]
    %SER 67.80 | %CER 9.63 [ 2288 / 23765, 119 ins, 1317 del, 852 sub ]
    %SER 32.25 | %CER 2.39 [ 3421 / 143203, 101 ins, 307 del, 3013 sub ]
    %SER 61.04 | %CER 8.65 [ 4458 / 51551, 729 ins, 1011 del, 2718 sub ]
    %SER 52.17 | %CER 5.72 [ 2119 / 37064, 84 ins, 73 del, 1962 sub ]
    %SER 61.94 | %CER 4.92 [ 1844 / 37506, 248 ins, 429 del, 1167 sub ]
    %SER 84.50 | %CER 8.80 [ 8217 / 93322, 459 ins, 3592 del, 4166 sub ]
    %SER 86.48 | %CER 21.29 [ 5604 / 26318, 473 ins, 820 del, 4311 sub ]
    %SER 91.56 | %CER 22.66 [ 4267 / 18832, 266 ins, 986 del, 3015 sub ]

WLM %SER 53.59 | %CER 6.93 [ 37189 / 536326, 2861 ins, 8139 del, 26189 sub ]        [-0.375, 0.5, 2.5]
    %SER 40.06 | %CER 5.63 [ 5894 / 104765, 638 ins, 109 del, 5147 sub ]
    %SER 66.55 | %CER 9.46 [ 2247 / 23765, 98 ins, 1306 del, 843 sub ]
    %SER 31.52 | %CER 2.31 [ 3304 / 143203, 96 ins, 299 del, 2909 sub ]
    %SER 59.81 | %CER 8.48 [ 4369 / 51551, 612 ins, 967 del, 2790 sub ]
    %SER 52.70 | %CER 5.59 [ 2072 / 37064, 82 ins, 74 del, 1916 sub ]
    %SER 62.32 | %CER 4.88 [ 1831 / 37506, 216 ins, 430 del, 1185 sub ]
    %SER 82.75 | %CER 8.41 [ 7845 / 93322, 423 ins, 3295 del, 4127 sub ]
    %SER 84.75 | %CER 20.84 [ 5485 / 26318, 456 ins, 728 del, 4301 sub ]
    %SER 90.78 | %CER 21.99 [ 4142 / 18832, 240 ins, 931 del, 2971 sub ]

DR  %SER 54.43 | %CER 7.13 [ 38220 / 536326, 3153 ins, 8651 del, 26416 sub ]        [0.0, 0.5, 4.5]

>>> + LM-trans-200m
SF  6.82
    %SER 36.34 | %CER 5.10 [ 5344 / 104765, 583 ins, 115 del, 4646 sub ]
    %SER 65.76 | %CER 9.72 [ 2310 / 23765, 109 ins, 1374 del, 827 sub ]
    %SER 29.43 | %CER 2.17 [ 3110 / 143203, 88 ins, 315 del, 2707 sub ]
    %SER 60.01 | %CER 8.45 [ 4355 / 51551, 731 ins, 1044 del, 2580 sub ]
    %SER 48.90 | %CER 5.29 [ 1961 / 37064, 75 ins, 77 del, 1809 sub ]
    %SER 60.11 | %CER 4.65 [ 1743 / 37506, 231 ins, 440 del, 1072 sub ]
    %SER 84.66 | %CER 8.71 [ 8129 / 93322, 430 ins, 3765 del, 3934 sub ]
    %SER 85.39 | %CER 20.73 [ 5455 / 26318, 449 ins, 830 del, 4176 sub ]
    %SER 91.04 | %CER 22.19 [ 4178 / 18832, 257 ins, 1004 del, 2917 sub ]

WLM 6.63
    %SER 35.80 | %CER 5.02 [ 5256 / 104765, 552 ins, 117 del, 4587 sub ]
    %SER 66.21 | %CER 9.59 [ 2279 / 23765, 100 ins, 1377 del, 802 sub ]
    %SER 28.29 | %CER 2.07 [ 2971 / 143203, 81 ins, 315 del, 2575 sub ]
    %SER 58.44 | %CER 8.26 [ 4256 / 51551, 596 ins, 1025 del, 2635 sub ]
    %SER 49.61 | %CER 5.16 [ 1911 / 37064, 66 ins, 75 del, 1770 sub ]
    %SER 59.57 | %CER 4.64 [ 1739 / 37506, 195 ins, 438 del, 1106 sub ]
    %SER 83.10 | %CER 8.29 [ 7732 / 93322, 389 ins, 3497 del, 3846 sub ]
    %SER 83.54 | %CER 20.29 [ 5341 / 26318, 429 ins, 762 del, 4150 sub ]
    %SER 89.74 | %CER 21.66 [ 4079 / 18832, 237 ins, 965 del, 2877 sub ]

DR
```

### Monitor figure
![monitor](./monitor.png)
