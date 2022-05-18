### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 91.27
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* trained on wenet speech M subset (1000 hour speech)

### Result
```
16 beam
dev             %SER 71.07 | %CER 11.16 [ 36887 / 330498, 1279 ins, 16227 del, 19381 sub ]
test_net        %SER 65.49 | %CER 12.76 [ 53035 / 415746, 1943 ins, 12961 del, 38131 sub ]
test_meeting    %SER 91.77 | %CER 20.97 [ 46214 / 220385, 1224 ins, 22925 del, 22065 sub ]
aishell-test    %SER 49.99 | %CER 7.22 [ 7560 / 104765, 249 ins, 201 del, 7110 sub ]

SF + lm-trans 0.25 3.0
dev     %SER 66.51 | %CER 9.33 [ 30828 / 330498, 3351 ins, 8563 del, 18914 sub ]
test_net        %SER 63.38 | %CER 12.22 [ 50795 / 415746, 5734 ins, 7268 del, 37793 sub ]
test_meeting    %SER 90.82 | %CER 18.42 [ 40596 / 220385, 5272 ins, 11091 del, 24233 sub ]
aishell-test    %SER 46.46 | %CER 6.83 [ 7156 / 104765, 831 ins, 99 del, 6226 sub ]

128 beam
dev             %SER 71.10 | %CER 11.14 [ 36833 / 330498, 1284 ins, 16210 del, 19339 sub ]
test_net        %SER 65.51 | %CER 12.75 [ 52991 / 415746, 1942 ins, 12914 del, 38135 sub ]
test_meeting    %SER 91.74 | %CER 20.88 [ 46025 / 220385, 1236 ins, 22703 del, 22086 sub ]
aishell-dev     %SER 45.05 | %CER 6.32 [ 12985 / 205341, 420 ins, 248 del, 12317 sub ]
aishell-test    %SER 49.97 | %CER 7.22 [ 7562 / 104765, 253 ins, 204 del, 7105 sub ]
combine-eval    %SER 60.67 | %CER 7.82 [ 41928 / 536326, 1314 ins, 11803 del, 28811 sub ]
# combine-eval: aishell-test + speechio 00-07
ILM: dev ppl 1150.63


+lm-aishell
aishell-dev     %SER 37.35 | %CER 5.11 [ 10497 / 205341, 401 ins, 347 del, 9749 sub ]   [0.5, 1.75]
aishell-test    %SER 39.53 | %CER 5.57 [ 5831 / 104765, 189 ins, 279 del, 5363 sub ]    [0.5, 1.5]

+lm-aishell-word
aishell-dev     %SER 41.10 | %CER 5.70 [ 11698 / 205341, 280 ins, 471 del, 10947 sub ]  [0.0, -2.0]
aishell-test    %SER 38.29 | %CER 5.37 [ 5625 / 104765, 141 ins, 450 del, 5034 sub ]    [0.53125, 0.75]

+lm-trans-word
dev             %SER 65.92 | %CER 9.38 [ 31004 / 330498, 2981 ins, 9406 del, 18617 sub ]        [0.25, 3.25]
test_net        %SER 62.44 | %CER 11.89 [ 49448 / 415746, 3341 ins, 9743 del, 36364 sub ]       [0.25, 2.0]
test_meeting    %SER 90.41 | %CER 18.58 [ 40937 / 220385, 2587 ins, 15868 del, 22482 sub ]      [0.1875, 2.0]
aishell-dev     
aishell-test    %SER 42.21 | %CER 5.99 [ 6273 / 104765, 185 ins, 398 del, 5690 sub ]    [0.5, 0.75]

+lm-trans
dev             %SER 66.66 | %CER 9.47 [ 31302 / 330498, 2967 ins, 9483 del, 18852 sub ]        [0.21875, 3.0]
test_net        %SER 63.22 | %CER 12.04 [ 50037 / 415746, 3080 ins, 9962 del, 36995 sub ]       [0.15625, 1.5]
test_meeting    %SER 90.61 | %CER 18.59 [ 40972 / 220385, 2592 ins, 15902 del, 22478 sub ]      [0.1875, 2.0]
aishell-dev     
aishell-test    %SER 43.67 | %CER 6.09 [ 6383 / 104765, 227 ins, 237 del, 5919 sub ]    [0.3125, 1.0]

+lm-nn-trans
dev             %SER 67.59 | %CER 9.64 [ 31848 / 330498, 2849 ins, 9527 del, 19472 sub ]        [0.1875, 2.75]
test_net        %SER 63.62 | %CER 12.18 [ 50636 / 415746, 3349 ins, 9582 del, 37705 sub ]       [0.1875, 1.75]
test_meeting    %SER 90.74 | %CER 18.66 [ 41121 / 220385, 2643 ins, 15774 del, 22704 sub ]      [0.1875, 2.0]
aishell-dev     
aishell-test    %SER 46.89 | %CER 6.67 [ 6988 / 104765, 213 ins, 259 del, 6516 sub ]    [0.25, 0.5]
```


in-domain LM integration + in-domain corpus
>>> SF
    %SER 65.03 | %CER 9.19 [ 30382 / 330498, 2910 ins, 9597 del, 17875 sub ]        [0.25, 3.125]
    %SER 61.66 | %CER 11.73 [ 48767 / 415746, 5030 ins, 7842 del, 35895 sub ]
    %SER 90.19 | %CER 18.36 [ 40467 / 220385, 3437 ins, 14598 del, 22432 sub ]

>>> DR      ppl: 37.89 
    %SER 65.03 | %CER 9.19 [ 30382 / 330498, 2910 ins, 9597 del, 17875 sub ]        [0.0, 0.25, 3.125]
    %SER 61.66 | %CER 11.73 [ 48767 / 415746, 5030 ins, 7842 del, 35895 sub ]
    %SER 90.19 | %CER 18.36 [ 40467 / 220385, 3437 ins, 14598 del, 22432 sub ]

>>> ILME    ppl: 94.32
    %SER 64.64 | %CER 9.10 [ 30091 / 330498, 2800 ins, 9947 del, 17344 sub ]        [-0.125, 0.375, 3.0]
    %SER 61.26 | %CER 11.56 [ 48049 / 415746, 4880 ins, 8076 del, 35093 sub ]
    %SER 90.00 | %CER 18.26 [ 40253 / 220385, 3246 ins, 14945 del, 22062 sub ]

>>> WLM     ppl: 79.33
    %SER 64.53 | %CER 9.07 [ 29962 / 330498, 2952 ins, 9665 del, 17345 sub ]        [-0.125, 0.375, 3.125]
    %SER 60.99 | %CER 11.54 [ 47965 / 415746, 5253 ins, 7840 del, 34872 sub ]
    %SER 89.96 | %CER 18.23 [ 40184 / 220385, 3377 ins, 14744 del, 22063 sub ]

LM adaptation

```
>>> w/o LM
    %SER 45.05 | %CER 6.32 [ 12985 / 205341, 420 ins, 248 del, 12317 sub ]
    %SER 49.97 | %CER 7.22 [ 7562 / 104765, 253 ins, 204 del, 7105 sub ]

>>> + lm-aishell, tuned on aishell-dev

SF  %SER 37.21 | %CER 5.11 [ 10488 / 205341, 343 ins, 411 del, 9734 sub ]   [0.5, 1.375]
    %SER 39.52 | %CER 5.56 [ 5828 / 104765, 180 ins, 294 del, 5354 sub ]

ILME  %SER 36.99 | %CER 4.99 [ 10251 / 205341, 351 ins, 315 del, 9585 sub ]   [-0.125, 0.5, 1.125]
      %SER 39.63 | %CER 5.55 [ 5816 / 104765, 191 ins, 244 del, 5381 sub ]

DR  %SER 37.60 | %CER 5.10 [ 10467 / 205341, 376 ins, 292 del, 9799 sub ]   [-0.125, 0.5, 1.375]
    %SER 40.29 | %CER 5.65 [ 5918 / 104765, 210 ins, 223 del, 5485 sub ]

WLM trained on transcript
    %SER 35.64 | %CER 4.76 [ 9775 / 205341, 367 ins, 314 del, 9094 sub ]    [-0.375, 0.625, 0.375]
    %SER 38.61 | %CER 5.33 [ 5579 / 104765, 196 ins, 233 del, 5150 sub ]

WLM trained on cc100 1mil
    %SER 35.68 | %CER 4.79 [ 9839 / 205341, 310 ins, 349 del, 9180 sub ]    [-0.25, 0.5, 0.125]

WLM 5-100
    %SER 35.63 | %CER 4.76 [ 9772 / 205341, 366 ins, 314 del, 9092 sub ]    [-0.375, 0.625, 0.375]
    %SER 38.61 | %CER 5.33 [ 5579 / 104765, 196 ins, 233 del, 5150 sub ]

WLM 5-500   
    %SER 36.24 | %CER 4.90 [ 10057 / 205341, 397 ins, 265 del, 9395 sub ]   [-0.25, 0.5, 0.75]
    %SER 39.07 | %CER 5.43 [ 5688 / 104765, 235 ins, 193 del, 5260 sub ]

WLM 5-1000  
    %SER 36.61 | %CER 4.98 [ 10227 / 205341, 352 ins, 339 del, 9536 sub ]   [-0.125, 0.5, 0.875]
    %SER 39.19 | %CER 5.49 [ 5752 / 104765, 192 ins, 251 del, 5309 sub ]

WLM 5-1500  
    %SER 36.70 | %CER 5.00 [ 10274 / 205341, 357 ins, 345 del, 9572 sub ]   [-0.125, 0.5, 0.875]
    %SER 39.17 | %CER 5.51 [ 5768 / 104765, 200 ins, 253 del, 5315 sub ]

WLM 5-2000  
    %SER 36.82 | %CER 5.02 [ 10298 / 205341, 365 ins, 341 del, 9592 sub ]   [-0.125, 0.5, 0.875]
    %SER 39.28 | %CER 5.52 [ 5786 / 104765, 200 ins, 256 del, 5330 sub ]

```

LM integration

```
>>> w/o LM
    %SER 49.97 | %CER 7.22 [ 7562 / 104765, 253 ins, 204 del, 7105 sub ]
    %SER 69.62 | %CER 11.27 [ 2679 / 23765, 48 ins, 1826 del, 805 sub ]
    %SER 46.95 | %CER 3.54 [ 5063 / 143203, 63 ins, 460 del, 4540 sub ]
    %SER 58.47 | %CER 8.16 [ 4209 / 51551, 279 ins, 1436 del, 2494 sub ]
    %SER 64.29 | %CER 7.17 [ 2657 / 37064, 40 ins, 102 del, 2515 sub ]
    %SER 63.08 | %CER 5.28 [ 1980 / 37506, 70 ins, 681 del, 1229 sub ]
    %SER 84.85 | %CER 9.18 [ 8566 / 93322, 202 ins, 4547 del, 3817 sub ]
    %SER 83.54 | %CER 19.92 [ 5243 / 26318, 220 ins, 1333 del, 3690 sub ]
    %SER 91.69 | %CER 21.08 [ 3969 / 18832, 139 ins, 1214 del, 2616 sub ]

>>> + lm-trans, tuned on combine-set
SF  %SER 56.63 | %CER 7.04 [ 37744 / 536326, 2607 ins, 8311 del, 26826 sub ]        [0.25, 2.375]
    %SER 44.96 | %CER 6.40 [ 6706 / 104765, 506 ins, 123 del, 6077 sub ]
    %SER 66.21 | %CER 8.66 [ 2059 / 23765, 105 ins, 1161 del, 793 sub ]
    %SER 42.24 | %CER 3.12 [ 4468 / 143203, 147 ins, 299 del, 4022 sub ]
    %SER 55.50 | %CER 7.38 [ 3806 / 51551, 551 ins, 910 del, 2345 sub ]
    %SER 58.41 | %CER 6.48 [ 2401 / 37064, 103 ins, 76 del, 2222 sub ]
    %SER 60.34 | %CER 4.84 [ 1817 / 37506, 143 ins, 530 del, 1144 sub ]
    %SER 81.16 | %CER 8.08 [ 7543 / 93322, 352 ins, 3477 del, 3714 sub ]
    %SER 82.58 | %CER 19.27 [ 5071 / 26318, 425 ins, 818 del, 3828 sub ]
    %SER 90.65 | %CER 20.57 [ 3873 / 18832, 275 ins, 917 del, 2681 sub ]
ILME  %SER 56.83 | %CER 7.10 [ 38061 / 536326, 2584 ins, 8886 del, 26591 sub ]        [-0.25, 0.5, 2.25]
    %SER 44.70 | %CER 6.36 [ 6667 / 104765, 543 ins, 130 del, 5994 sub ]
    %SER 66.78 | %CER 8.94 [ 2125 / 23765, 97 ins, 1248 del, 780 sub ]
    %SER 42.08 | %CER 3.06 [ 4386 / 143203, 155 ins, 314 del, 3917 sub ]
    %SER 56.16 | %CER 7.54 [ 3889 / 51551, 553 ins, 963 del, 2373 sub ]
    %SER 58.11 | %CER 6.33 [ 2347 / 37064, 110 ins, 75 del, 2162 sub ]
    %SER 60.56 | %CER 4.92 [ 1845 / 37506, 141 ins, 554 del, 1150 sub ]
    %SER 82.59 | %CER 8.37 [ 7812 / 93322, 329 ins, 3750 del, 3733 sub ]
    %SER 83.28 | %CER 19.21 [ 5055 / 26318, 395 ins, 886 del, 3774 sub ]
    %SER 90.00 | %CER 20.90 [ 3935 / 18832, 261 ins, 966 del, 2708 sub ]

DR  %SER 56.63 | %CER 7.04 [ 37744 / 536326, 2607 ins, 8311 del, 26826 sub ]        [0.0, 0.25, 2.375]
  same as SF

>>> + lm-vary/trans_5m
SF  %SER 51.74 | %CER 6.39 [ 34281 / 536326, 2705 ins, 8284 del, 23292 sub ]        [0.25, 2.625]
    %SER 38.50 | %CER 5.29 [ 5545 / 104765, 490 ins, 116 del, 4939 sub ]
    %SER 65.76 | %CER 8.62 [ 2049 / 23765, 104 ins, 1209 del, 736 sub ]
    %SER 32.61 | %CER 2.30 [ 3294 / 143203, 133 ins, 297 del, 2864 sub ]
    %SER 54.73 | %CER 7.23 [ 3726 / 51551, 600 ins, 880 del, 2246 sub ]
    %SER 49.67 | %CER 5.07 [ 1879 / 37064, 121 ins, 68 del, 1690 sub ]
    %SER 58.66 | %CER 4.67 [ 1750 / 37506, 159 ins, 528 del, 1063 sub ]
    %SER 80.21 | %CER 7.82 [ 7302 / 93322, 368 ins, 3526 del, 3408 sub ]
    %SER 81.55 | %CER 18.75 [ 4935 / 26318, 445 ins, 768 del, 3722 sub ]
    %SER 89.48 | %CER 20.18 [ 3801 / 18832, 285 ins, 892 del, 2624 sub ]

ILME  %SER 49.99 | %CER 6.25 [ 33541 / 536326, 2970 ins, 8518 del, 22053 sub ]        [-0.25, 0.5, 2.875]
    %SER 35.91 | %CER 4.95 [ 5189 / 104765, 613 ins, 104 del, 4472 sub ]
    %SER 65.30 | %CER 8.73 [ 2074 / 23765, 108 ins, 1247 del, 719 sub ]
    %SER 28.35 | %CER 2.02 [ 2888 / 143203, 148 ins, 307 del, 2433 sub ]
    %SER 54.89 | %CER 7.30 [ 3764 / 51551, 665 ins, 875 del, 2224 sub ]
    %SER 46.05 | %CER 4.56 [ 1689 / 37064, 123 ins, 63 del, 1503 sub ]
    %SER 58.58 | %CER 4.58 [ 1718 / 37506, 186 ins, 523 del, 1009 sub ]
    %SER 80.65 | %CER 7.97 [ 7439 / 93322, 382 ins, 3736 del, 3321 sub ]
    %SER 82.45 | %CER 18.76 [ 4937 / 26318, 459 ins, 771 del, 3707 sub ]
    %SER 90.00 | %CER 20.41 [ 3843 / 18832, 286 ins, 892 del, 2665 sub ]

DR  %SER 50.70 | %CER 6.37 [ 34148 / 536326, 2492 ins, 9282 del, 22374 sub ]        [-0.25, 0.5, 2.375]
    %SER 36.72 | %CER 5.03 [ 5267 / 104765, 490 ins, 126 del, 4651 sub ]
    %SER 66.21 | %CER 9.37 [ 2226 / 23765, 93 ins, 1417 del, 716 sub ]
    %SER 29.24 | %CER 2.08 [ 2979 / 143203, 122 ins, 326 del, 2531 sub ]
    %SER 54.89 | %CER 7.36 [ 3796 / 51551, 542 ins, 981 del, 2273 sub ]
    %SER 47.30 | %CER 4.68 [ 1736 / 37064, 105 ins, 69 del, 1562 sub ]
    %SER 59.34 | %CER 4.60 [ 1727 / 37506, 141 ins, 565 del, 1021 sub ]
    %SER 81.73 | %CER 8.11 [ 7566 / 93322, 346 ins, 3948 del, 3272 sub ]
    %SER 82.38 | %CER 19.04 [ 5011 / 26318, 404 ins, 890 del, 3717 sub ]
    %SER 90.00 | %CER 20.39 [ 3840 / 18832, 249 ins, 960 del, 2631 sub ]

WLM         %SER 51.41 | %CER 6.32 [ 33888 / 536326, 2599 ins, 8363 del, 22926 sub ]        [-0.25, 0.25, 1.375]
    %SER 38.60 | %CER 5.31 [ 5567 / 104765, 549 ins, 107 del, 4911 sub ]
    %SER 65.07 | %CER 9.08 [ 2157 / 23765, 90 ins, 1342 del, 725 sub ]
    %SER 31.94 | %CER 2.25 [ 3218 / 143203, 172 ins, 289 del, 2757 sub ]
    %SER 53.69 | %CER 7.04 [ 3631 / 51551, 507 ins, 900 del, 2224 sub ]
    %SER 49.91 | %CER 4.90 [ 1815 / 37064, 117 ins, 65 del, 1633 sub ]
    %SER 58.89 | %CER 4.72 [ 1770 / 37506, 160 ins, 531 del, 1079 sub ]
    %SER 79.67 | %CER 7.61 [ 7100 / 93322, 339 ins, 3483 del, 3278 sub ]
    %SER 80.97 | %CER 18.42 [ 4848 / 26318, 417 ins, 750 del, 3681 sub ]
    %SER 89.61 | %CER 20.08 [ 3782 / 18832, 248 ins, 896 del, 2638 sub ]


WLM cc100-1mil   %SER 49.67 | %CER 6.17 [ 33085 / 536326, 2906 ins, 8306 del, 21873 sub ]    [-0.375, 0.5, 1.875]

WLM 5-100   %SER 49.84 | %CER 6.19 [ 33178 / 536326, 2876 ins, 8654 del, 21648 sub ]        [-0.375, 0.5, 2.0]      5.88

WLM 5-500   %SER 50.28 | %CER 6.29 [ 33751 / 536326, 2823 ins, 8744 del, 22184 sub ]        [-0.375, 0.5, 1.875]    5.96

WLM 5-1000  %SER 50.24 | %CER 6.30 [ 33775 / 536326, 3022 ins, 8501 del, 22252 sub ]        [-0.25, 0.5, 2.625]     5.94

WLM 5-1500  %SER 50.62 | %CER 6.36 [ 34099 / 536326, 2865 ins, 8800 del, 22434 sub ]        [-0.375, 0.5, 1.75]     6.02

WLM 5-2000  %SER 50.61 | %CER 6.35 [ 34079 / 536326, 3058 ins, 8508 del, 22513 sub ]        [-0.375, 0.5, 1.875]    6.01


>>> + lm-vary/trans_10m
SF [0.25, 2.625]
    %SER 37.60 | %CER 5.18 [ 5428 / 104765, 476 ins, 119 del, 4833 sub ]
    %SER 65.64 | %CER 8.62 [ 2048 / 23765, 98 ins, 1226 del, 724 sub ]
    %SER 31.92 | %CER 2.23 [ 3193 / 143203, 128 ins, 298 del, 2767 sub ]
    %SER 54.49 | %CER 7.21 [ 3715 / 51551, 597 ins, 889 del, 2229 sub ]
    %SER 49.85 | %CER 5.08 [ 1884 / 37064, 111 ins, 71 del, 1702 sub ]
    %SER 57.82 | %CER 4.60 [ 1726 / 37506, 157 ins, 521 del, 1048 sub ]
    %SER 80.27 | %CER 7.78 [ 7261 / 93322, 355 ins, 3563 del, 3343 sub ]
    %SER 81.61 | %CER 18.67 [ 4913 / 26318, 442 ins, 782 del, 3689 sub ]
    %SER 89.61 | %CER 20.17 [ 3799 / 18832, 284 ins, 893 del, 2622 sub ]

ILME  [-0.25, 0.5, 2.875]
    %SER 34.87 | %CER 4.80 [ 5031 / 104765, 592 ins, 111 del, 4328 sub ]
    %SER 64.73 | %CER 8.73 [ 2074 / 23765, 117 ins, 1273 del, 684 sub ]
    %SER 27.78 | %CER 1.97 [ 2818 / 143203, 149 ins, 312 del, 2357 sub ]
    %SER 54.49 | %CER 7.21 [ 3715 / 51551, 656 ins, 895 del, 2164 sub ]
    %SER 45.93 | %CER 4.52 [ 1676 / 37064, 121 ins, 64 del, 1491 sub ]
    %SER 58.05 | %CER 4.53 [ 1700 / 37506, 186 ins, 529 del, 985 sub ]
    %SER 81.26 | %CER 7.95 [ 7416 / 93322, 376 ins, 3805 del, 3235 sub ]
    %SER 81.42 | %CER 18.55 [ 4883 / 26318, 454 ins, 784 del, 3645 sub ]
    %SER 89.48 | %CER 20.35 [ 3833 / 18832, 288 ins, 889 del, 2656 sub ]

DR  [-0.25, 0.5, 2.375]
    %SER 35.38 | %CER 4.85 [ 5079 / 104765, 483 ins, 133 del, 4463 sub ]
    %SER 65.98 | %CER 9.33 [ 2218 / 23765, 96 ins, 1429 del, 693 sub ]
    %SER 28.47 | %CER 2.01 [ 2884 / 143203, 119 ins, 330 del, 2435 sub ]
    %SER 54.96 | %CER 7.33 [ 3779 / 51551, 541 ins, 1013 del, 2225 sub ]
    %SER 46.88 | %CER 4.62 [ 1713 / 37064, 94 ins, 72 del, 1547 sub ]
    %SER 58.81 | %CER 4.59 [ 1723 / 37506, 146 ins, 566 del, 1011 sub ]
    %SER 81.93 | %CER 8.09 [ 7550 / 93322, 338 ins, 4013 del, 3199 sub ]
    %SER 81.87 | %CER 18.86 [ 4964 / 26318, 388 ins, 889 del, 3687 sub ]
    %SER 89.48 | %CER 20.36 [ 3834 / 18832, 238 ins, 958 del, 2638 sub ]

WLM 6.27
    %SER 37.81 | %CER 5.20 [ 5443 / 104765, 554 ins, 109 del, 4780 sub ]
    %SER 65.53 | %CER 9.07 [ 2156 / 23765, 88 ins, 1345 del, 723 sub ]
    %SER 31.52 | %CER 2.20 [ 3156 / 143203, 164 ins, 294 del, 2698 sub ]
    %SER 53.66 | %CER 7.03 [ 3624 / 51551, 501 ins, 915 del, 2208 sub ]
    %SER 49.79 | %CER 4.93 [ 1828 / 37064, 116 ins, 68 del, 1644 sub ]
    %SER 58.58 | %CER 4.67 [ 1752 / 37506, 159 ins, 529 del, 1064 sub ]
    %SER 79.67 | %CER 7.57 [ 7068 / 93322, 327 ins, 3515 del, 3226 sub ]
    %SER 80.59 | %CER 18.36 [ 4831 / 26318, 424 ins, 755 del, 3652 sub ]
    %SER 88.83 | %CER 19.93 [ 3754 / 18832, 240 ins, 894 del, 2620 sub ]

>>> + lm-vary/trans_20m
SF [0.25, 2.625]
    %SER 36.80 | %CER 5.07 [ 5315 / 104765, 480 ins, 121 del, 4714 sub ]
    %SER 65.30 | %CER 8.68 [ 2062 / 23765, 96 ins, 1238 del, 728 sub ]
    %SER 31.43 | %CER 2.19 [ 3131 / 143203, 129 ins, 302 del, 2700 sub ]
    %SER 54.26 | %CER 7.09 [ 3656 / 51551, 581 ins, 891 del, 2184 sub ]
    %SER 48.37 | %CER 4.88 [ 1808 / 37064, 104 ins, 69 del, 1635 sub ]
    %SER 57.44 | %CER 4.56 [ 1709 / 37506, 153 ins, 528 del, 1028 sub ]
    %SER 80.08 | %CER 7.74 [ 7221 / 93322, 349 ins, 3571 del, 3301 sub ]
    %SER 81.04 | %CER 18.47 [ 4862 / 26318, 437 ins, 784 del, 3641 sub ]
    %SER 89.35 | %CER 20.05 [ 3775 / 18832, 284 ins, 894 del, 2597 sub ]

ILME  [-0.25, 0.5, 2.875]
    %SER 34.04 | %CER 4.68 [ 4907 / 104765, 596 ins, 110 del, 4201 sub ]
    %SER 64.39 | %CER 8.78 [ 2086 / 23765, 111 ins, 1288 del, 687 sub ]
    %SER 27.32 | %CER 1.95 [ 2791 / 143203, 150 ins, 306 del, 2335 sub ]
    %SER 54.76 | %CER 7.15 [ 3684 / 51551, 648 ins, 895 del, 2141 sub ]
    %SER 44.62 | %CER 4.26 [ 1579 / 37064, 112 ins, 65 del, 1402 sub ]
    %SER 57.51 | %CER 4.52 [ 1694 / 37506, 185 ins, 532 del, 977 sub ]
    %SER 80.94 | %CER 7.88 [ 7356 / 93322, 361 ins, 3845 del, 3150 sub ]
    %SER 81.68 | %CER 18.43 [ 4850 / 26318, 450 ins, 790 del, 3610 sub ]
    %SER 89.61 | %CER 20.26 [ 3815 / 18832, 288 ins, 897 del, 2630 sub ]

DR  [-0.25, 0.5, 2.375]
    %SER 34.45 | %CER 4.69 [ 4918 / 104765, 453 ins, 133 del, 4332 sub ]
    %SER 65.19 | %CER 9.51 [ 2260 / 23765, 96 ins, 1476 del, 688 sub ]
    %SER 27.72 | %CER 1.98 [ 2836 / 143203, 114 ins, 334 del, 2388 sub ]
    %SER 54.59 | %CER 7.18 [ 3700 / 51551, 537 ins, 997 del, 2166 sub ]
    %SER 46.11 | %CER 4.47 [ 1656 / 37064, 90 ins, 70 del, 1496 sub ]
    %SER 58.28 | %CER 4.54 [ 1704 / 37506, 144 ins, 572 del, 988 sub ]
    %SER 81.77 | %CER 8.02 [ 7485 / 93322, 323 ins, 4030 del, 3132 sub ]
    %SER 81.55 | %CER 18.75 [ 4935 / 26318, 392 ins, 896 del, 3647 sub ]
    %SER 89.61 | %CER 20.23 [ 3809 / 18832, 246 ins, 962 del, 2601 sub ]

WLM 6.20
    %SER 37.11 | %CER 5.08 [ 5322 / 104765, 549 ins, 112 del, 4661 sub ]
    %SER 64.51 | %CER 9.05 [ 2151 / 23765, 86 ins, 1351 del, 714 sub ]
    %SER 30.70 | %CER 2.16 [ 3093 / 143203, 162 ins, 295 del, 2636 sub ]
    %SER 53.66 | %CER 6.97 [ 3593 / 51551, 496 ins, 919 del, 2178 sub ]
    %SER 48.78 | %CER 4.78 [ 1772 / 37064, 115 ins, 68 del, 1589 sub ]
    %SER 58.12 | %CER 4.61 [ 1730 / 37506, 161 ins, 529 del, 1040 sub ]
    %SER 79.76 | %CER 7.54 [ 7034 / 93322, 325 ins, 3531 del, 3178 sub ]
    %SER 80.72 | %CER 18.35 [ 4830 / 26318, 423 ins, 770 del, 3637 sub ]
    %SER 88.57 | %CER 19.85 [ 3738 / 18832, 243 ins, 889 del, 2606 sub ]


>>> + lm-vary/trans_40m
SF [0.25, 2.625]
    %SER 36.30 | %CER 4.97 [ 5212 / 104765, 479 ins, 118 del, 4615 sub ]
    %SER 63.94 | %CER 8.64 [ 2053 / 23765, 93 ins, 1263 del, 697 sub ]
    %SER 30.70 | %CER 2.15 [ 3074 / 143203, 126 ins, 301 del, 2647 sub ]
    %SER 54.49 | %CER 7.11 [ 3663 / 51551, 582 ins, 916 del, 2165 sub ]
    %SER 47.47 | %CER 4.80 [ 1780 / 37064, 108 ins, 68 del, 1604 sub ]
    %SER 57.67 | %CER 4.53 [ 1698 / 37506, 155 ins, 536 del, 1007 sub ]
    %SER 80.02 | %CER 7.73 [ 7217 / 93322, 356 ins, 3595 del, 3266 sub ]
    %SER 80.91 | %CER 18.50 [ 4869 / 26318, 440 ins, 782 del, 3647 sub ]
    %SER 89.35 | %CER 20.00 [ 3767 / 18832, 285 ins, 906 del, 2576 sub ]

ILME  [-0.25, 0.5, 2.875]
    %SER 33.26 | %CER 4.53 [ 4750 / 104765, 556 ins, 107 del, 4087 sub ]
    %SER 64.05 | %CER 8.74 [ 2078 / 23765, 105 ins, 1304 del, 669 sub ]
    %SER 26.61 | %CER 1.90 [ 2722 / 143203, 134 ins, 310 del, 2278 sub ]
    %SER 54.66 | %CER 7.07 [ 3646 / 51551, 635 ins, 916 del, 2095 sub ]
    %SER 44.21 | %CER 4.22 [ 1565 / 37064, 118 ins, 64 del, 1383 sub ]
    %SER 57.82 | %CER 4.47 [ 1677 / 37506, 182 ins, 530 del, 965 sub ]
    %SER 81.10 | %CER 7.83 [ 7311 / 93322, 368 ins, 3873 del, 3070 sub ]
    %SER 81.74 | %CER 18.31 [ 4819 / 26318, 464 ins, 787 del, 3568 sub ]
    %SER 89.22 | %CER 20.20 [ 3805 / 18832, 290 ins, 910 del, 2605 sub ]

DR  [-0.25, 0.5, 2.375]
    %SER 33.79 | %CER 4.60 [ 4819 / 104765, 436 ins, 126 del, 4257 sub ]
    %SER 64.28 | %CER 9.43 [ 2242 / 23765, 85 ins, 1496 del, 661 sub ]
    %SER 27.03 | %CER 1.93 [ 2765 / 143203, 106 ins, 331 del, 2328 sub ]
    %SER 54.99 | %CER 7.21 [ 3715 / 51551, 524 ins, 1030 del, 2161 sub ]
    %SER 45.45 | %CER 4.34 [ 1608 / 37064, 88 ins, 72 del, 1448 sub ]
    %SER 57.89 | %CER 4.52 [ 1696 / 37506, 137 ins, 570 del, 989 sub ]
    %SER 81.61 | %CER 7.96 [ 7433 / 93322, 317 ins, 4083 del, 3033 sub ]
    %SER 81.36 | %CER 18.58 [ 4891 / 26318, 393 ins, 897 del, 3601 sub ]
    %SER 90.00 | %CER 20.22 [ 3808 / 18832, 243 ins, 980 del, 2585 sub ]

WLM
    %SER 36.40 | %CER 5.00 [ 5243 / 104765, 542 ins, 106 del, 4595 sub ]
    %SER 64.05 | %CER 9.03 [ 2145 / 23765, 87 ins, 1371 del, 687 sub ]
    %SER 29.79 | %CER 2.10 [ 3006 / 143203, 154 ins, 296 del, 2556 sub ]
    %SER 53.56 | %CER 6.90 [ 3558 / 51551, 484 ins, 929 del, 2145 sub ]
    %SER 47.83 | %CER 4.67 [ 1730 / 37064, 109 ins, 66 del, 1555 sub ]
    %SER 57.74 | %CER 4.57 [ 1715 / 37506, 155 ins, 533 del, 1027 sub ]
    %SER 79.73 | %CER 7.51 [ 7012 / 93322, 320 ins, 3561 del, 3131 sub ]
    %SER 80.40 | %CER 18.19 [ 4788 / 26318, 413 ins, 753 del, 3622 sub ]
    %SER 88.96 | %CER 19.80 [ 3729 / 18832, 244 ins, 898 del, 2587 sub ]

>>> + lm-vary/trans_80m
SF [0.25, 2.625]
    %SER 35.56 | %CER 4.87 [ 5099 / 104765, 471 ins, 116 del, 4512 sub ]
    %SER 64.05 | %CER 8.63 [ 2050 / 23765, 93 ins, 1261 del, 696 sub ]
    %SER 30.10 | %CER 2.10 [ 3010 / 143203, 119 ins, 301 del, 2590 sub ]
    %SER 54.56 | %CER 7.06 [ 3640 / 51551, 578 ins, 910 del, 2152 sub ]
    %SER 47.00 | %CER 4.74 [ 1757 / 37064, 104 ins, 63 del, 1590 sub ]
    %SER 57.36 | %CER 4.47 [ 1678 / 37506, 157 ins, 537 del, 984 sub ]
    %SER 79.76 | %CER 7.72 [ 7209 / 93322, 351 ins, 3620 del, 3238 sub ]
    %SER 80.85 | %CER 18.35 [ 4830 / 26318, 433 ins, 781 del, 3616 sub ]
    %SER 88.83 | %CER 19.91 [ 3750 / 18832, 279 ins, 907 del, 2564 sub ]

ILME  [-0.25, 0.5, 2.875]
    %SER 32.27 | %CER 4.40 [ 4614 / 104765, 557 ins, 108 del, 3949 sub ]
    %SER 63.71 | %CER 8.87 [ 2108 / 23765, 108 ins, 1331 del, 669 sub ]
    %SER 26.08 | %CER 1.86 [ 2669 / 143203, 131 ins, 315 del, 2223 sub ]
    %SER 54.16 | %CER 7.08 [ 3649 / 51551, 652 ins, 915 del, 2082 sub ]
    %SER 43.49 | %CER 4.17 [ 1547 / 37064, 119 ins, 64 del, 1364 sub ]
    %SER 57.06 | %CER 4.43 [ 1660 / 37506, 183 ins, 537 del, 940 sub ]
    %SER 81.19 | %CER 7.85 [ 7326 / 93322, 373 ins, 3897 del, 3056 sub ]
    %SER 80.97 | %CER 18.10 [ 4764 / 26318, 451 ins, 791 del, 3522 sub ]
    %SER 88.31 | %CER 20.11 [ 3788 / 18832, 292 ins, 912 del, 2584 sub ]

DR  [-0.25, 0.5, 2.375]
    %SER 32.90 | %CER 4.47 [ 4680 / 104765, 424 ins, 129 del, 4127 sub ]
    %SER 64.16 | %CER 9.39 [ 2232 / 23765, 80 ins, 1505 del, 647 sub ]
    %SER 26.85 | %CER 1.91 [ 2742 / 143203, 104 ins, 339 del, 2299 sub ]
    %SER 54.36 | %CER 7.09 [ 3654 / 51551, 519 ins, 1036 del, 2099 sub ]
    %SER 44.50 | %CER 4.22 [ 1564 / 37064, 84 ins, 69 del, 1411 sub ]
    %SER 57.67 | %CER 4.50 [ 1687 / 37506, 147 ins, 575 del, 965 sub ]
    %SER 81.70 | %CER 7.97 [ 7441 / 93322, 321 ins, 4117 del, 3003 sub ]
    %SER 80.91 | %CER 18.58 [ 4891 / 26318, 393 ins, 916 del, 3582 sub ]
    %SER 89.22 | %CER 20.17 [ 3799 / 18832, 243 ins, 987 del, 2569 sub ]

WLM
    %SER 35.67 | %CER 4.91 [ 5145 / 104765, 548 ins, 109 del, 4488 sub ]
    %SER 64.39 | %CER 9.11 [ 2166 / 23765, 89 ins, 1387 del, 690 sub ]
    %SER 29.20 | %CER 2.06 [ 2949 / 143203, 146 ins, 295 del, 2508 sub ]
    %SER 53.26 | %CER 6.83 [ 3521 / 51551, 478 ins, 923 del, 2120 sub ]
    %SER 47.89 | %CER 4.68 [ 1734 / 37064, 113 ins, 66 del, 1555 sub ]
    %SER 57.74 | %CER 4.52 [ 1697 / 37506, 150 ins, 530 del, 1017 sub ]
    %SER 79.42 | %CER 7.52 [ 7018 / 93322, 323 ins, 3570 del, 3125 sub ]
    %SER 80.46 | %CER 18.14 [ 4773 / 26318, 415 ins, 764 del, 3594 sub ]
    %SER 87.66 | %CER 19.70 [ 3710 / 18832, 239 ins, 902 del, 2569 sub ]


>>> + lm-vary/trans_200m
SF [0.25, 2.625]
    %SER 34.57 | %CER 4.71 [ 4938 / 104765, 460 ins, 116 del, 4362 sub ]
    %SER 63.59 | %CER 8.59 [ 2042 / 23765, 90 ins, 1263 del, 689 sub ]
    %SER 29.91 | %CER 2.07 [ 2971 / 143203, 109 ins, 301 del, 2561 sub ]
    %SER 54.29 | %CER 7.02 [ 3617 / 51551, 580 ins, 923 del, 2114 sub ]
    %SER 46.35 | %CER 4.61 [ 1707 / 37064, 95 ins, 64 del, 1548 sub ]
    %SER 57.44 | %CER 4.47 [ 1677 / 37506, 167 ins, 538 del, 972 sub ]
    %SER 79.83 | %CER 7.68 [ 7167 / 93322, 345 ins, 3626 del, 3196 sub ]
    %SER 81.10 | %CER 18.26 [ 4806 / 26318, 428 ins, 786 del, 3592 sub ]
    %SER 88.83 | %CER 19.87 [ 3742 / 18832, 280 ins, 904 del, 2558 sub ]

ILME  [-0.25, 0.5, 2.875]
    %SER 31.30 | %CER 4.26 [ 4468 / 104765, 540 ins, 110 del, 3818 sub ]
    %SER 63.59 | %CER 8.90 [ 2114 / 23765, 106 ins, 1357 del, 651 sub ]
    %SER 25.63 | %CER 1.83 [ 2622 / 143203, 124 ins, 313 del, 2185 sub ]
    %SER 53.69 | %CER 6.96 [ 3590 / 51551, 651 ins, 913 del, 2026 sub ]
    %SER 42.60 | %CER 4.09 [ 1515 / 37064, 113 ins, 62 del, 1340 sub ]
    %SER 56.60 | %CER 4.40 [ 1649 / 37506, 185 ins, 550 del, 914 sub ]
    %SER 80.88 | %CER 7.82 [ 7301 / 93322, 368 ins, 3925 del, 3008 sub ]
    %SER 80.65 | %CER 17.96 [ 4727 / 26318, 439 ins, 793 del, 3495 sub ]
    %SER 88.70 | %CER 19.82 [ 3732 / 18832, 286 ins, 918 del, 2528 sub ]

DR  [-0.25, 0.5, 2.375]
    %SER 31.38 | %CER 4.24 [ 4447 / 104765, 401 ins, 127 del, 3919 sub ]
    %SER 64.62 | %CER 9.44 [ 2243 / 23765, 79 ins, 1514 del, 650 sub ]
    %SER 26.16 | %CER 1.86 [ 2661 / 143203, 97 ins, 336 del, 2228 sub ]
    %SER 53.99 | %CER 7.01 [ 3616 / 51551, 523 ins, 1056 del, 2037 sub ]
    %SER 43.49 | %CER 4.13 [ 1531 / 37064, 85 ins, 69 del, 1377 sub ]
    %SER 56.83 | %CER 4.42 [ 1656 / 37506, 144 ins, 579 del, 933 sub ]
    %SER 81.39 | %CER 7.94 [ 7410 / 93322, 320 ins, 4136 del, 2954 sub ]
    %SER 81.42 | %CER 18.52 [ 4873 / 26318, 390 ins, 933 del, 3550 sub ]
    %SER 89.09 | %CER 19.95 [ 3757 / 18832, 243 ins, 983 del, 2531 sub ]

WLM
    6.02
    %SER 34.70 | %CER 4.73 [ 4960 / 104765, 517 ins, 110 del, 4333 sub ]
    %SER 64.28 | %CER 9.13 [ 2169 / 23765, 83 ins, 1406 del, 680 sub ]
    %SER 28.94 | %CER 2.03 [ 2901 / 143203, 142 ins, 295 del, 2464 sub ]
    %SER 53.06 | %CER 6.79 [ 3499 / 51551, 481 ins, 932 del, 2086 sub ]
    %SER 47.12 | %CER 4.60 [ 1706 / 37064, 111 ins, 66 del, 1529 sub ]
    %SER 56.90 | %CER 4.45 [ 1668 / 37506, 146 ins, 530 del, 992 sub ]
    %SER 79.45 | %CER 7.46 [ 6959 / 93322, 318 ins, 3583 del, 3058 sub ]
    %SER 80.20 | %CER 18.01 [ 4739 / 26318, 407 ins, 761 del, 3571 sub ]
    %SER 87.53 | %CER 19.57 [ 3686 / 18832, 241 ins, 903 del, 2542 sub ]
```

```
+ cheat-lm
%SER 30.78 | %CER 3.67 [ 19696 / 536326, 1173 ins, 6788 del, 11735 sub ]        [1.65625, 7.0]
aishell-test          %SER 18.92 | %CER 2.35 [ 2465 / 104765, 94 ins, 150 del, 2221 sub ]     [1.46875, 2.75]
speechio_asr_zh00000  %SER 45.96 | %CER 5.64 [ 1341 / 23765, 60 ins, 906 del, 375 sub ]       [0.6875, 4.5]
speechio_asr_zh00001  %SER 17.18 | %CER 1.23 [ 1767 / 143203, 30 ins, 336 del, 1401 sub ]     [1.375, 1.75]
speechio_asr_zh00002  %SER 26.93 | %CER 3.35 [ 1725 / 51551, 200 ins, 696 del, 829 sub ]      [1.3125, 4.5]
speechio_asr_zh00003  %SER 27.81 | %CER 2.60 [ 964 / 37064, 22 ins, 82 del, 860 sub ]         [1.75, 2.5]
speechio_asr_zh00004  %SER 30.97 | %CER 2.24 [ 841 / 37506, 40 ins, 380 del, 421 sub ]        [1.46875, 5.0]
speechio_asr_zh00005  %SER 55.21 | %CER 4.56 [ 4251 / 93322, 237 ins, 2328 del, 1686 sub ]    [1.5, 8.75]
speechio_asr_zh00006  %SER 62.08 | %CER 12.92 [ 3399 / 26318, 224 ins, 942 del, 2233 sub ]    [0.5625, 2.75]
speechio_asr_zh00007  %SER 76.49 | %CER 15.89 [ 2992 / 18832, 112 ins, 1070 del, 1810 sub ]   [0.5625, 1.75]
```


### Monitor figure
![monitor](./monitor.png)