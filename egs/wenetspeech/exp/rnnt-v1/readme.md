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
 
128 beam
dev             %SER 71.10 | %CER 11.14 [ 36833 / 330498, 1284 ins, 16210 del, 19339 sub ]
test_net        %SER 65.51 | %CER 12.75 [ 52991 / 415746, 1942 ins, 12914 del, 38135 sub ]
test_meeting    %SER 91.74 | %CER 20.88 [ 46025 / 220385, 1236 ins, 22703 del, 22086 sub ]
aishell-dev     %SER 45.05 | %CER 6.32 [ 12985 / 205341, 420 ins, 248 del, 12317 sub ]
aishell-test    %SER 49.97 | %CER 7.22 [ 7562 / 104765, 253 ins, 204 del, 7105 sub ]

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

```
beam 128
density ratio lm-trans lm-aishell beta=0
%SER 40.09 | %CER 5.70 [ 5971 / 104765, 126 ins, 503 del, 5342 sub ]    [-0.03125, 0.5]
```

LM integration

```
oracle
%SER 40.51 | %CER 5.50 [ 18163 / 330498, 501 ins, 8594 del, 9068 sub ]
nolm
%SER 71.10 | %CER 11.14 [ 36833 / 330498, 1284 ins, 16210 del, 19339 sub ]

+ trans
%SER 66.66 | %CER 9.47 [ 31302 / 330498, 2967 ins, 9483 del, 18852 sub ]        [0.21875, 3.0]

+ trans-5m
%SER 64.77 | %CER 9.14 [ 30221 / 330498, 2941 ins, 9782 del, 17498 sub ]        [0.25, 3.25]

+ trans-10m
%SER 64.52 | %CER 9.11 [ 30104 / 330498, 3026 ins, 9851 del, 17227 sub ]        [0.28125, 3.5]

+ trans-15m
%SER 64.51 | %CER 9.07 [ 29992 / 330498, 2903 ins, 9903 del, 17186 sub ]        [0.25, 3.25]

+ trans-20m
%SER 64.77 | %CER 9.14 [ 30221 / 330498, 2941 ins, 9782 del, 17498 sub ]        [0.25, 3.25]

+ trans-25m
%SER 64.77 | %CER 9.14 [ 30221 / 330498, 2941 ins, 9782 del, 17498 sub ]        [0.25, 3.25]

+ trans-30m
%SER 64.14 | %CER 9.01 [ 29787 / 330498, 3176 ins, 9617 del, 16994 sub ]        [0.28125, 3.75]

+ trans-35m
%SER 64.13 | %CER 9.02 [ 29803 / 330498, 3180 ins, 9622 del, 17001 sub ]        [0.28125, 3.75]

+trans-40m
%SER 64.08 | %CER 9.01 [ 29786 / 330498, 2972 ins, 9926 del, 16888 sub ]        [0.28125, 3.5]
```

### Monitor figure
![monitor](./monitor.png)
