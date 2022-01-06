### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 91.27
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* trained on wenet speech M subset (1000 hour speech)

### Result
```
no lm
dev     %SER 71.07 | %CER 11.16 [36887 / 330498, 1279 ins, 16227 del, 19381 sub ]
test_meeting  %SER 91.77 | %CER 20.97 [46214 / 220385, 1224 ins, 22925 del, 22065 sub ]
test_net      %SER 65.49 | %CER 12.76 [53035 / 415746, 1943 ins, 12961 del, 38131 sub ]
aishell-test    %SER 49.99 | %CER 7.22 [7560 / 104765, 249 ins, 201 del, 7110 sub ] 

0.25 3.53125
dev     %SER 67.00 | %CER 9.80 [32384 / 330498, 2439 ins, 11058 del, 18887 sub ]
test_meeting    %SER 91.17 | %CER 19.44 [42845 / 220385, 2662 ins, 17412 del, 22771 sub ]
test_net        %SER 63.70 | %CER 12.26 [50975 / 415746, 4384 ins, 8751 del, 37840 sub ]
aishell-test    %SER 47.41 | %CER 6.78 [7105 / 104765, 728 ins, 95 del, 6282 sub ]

0.25 2.5
dev     %SER 67.25 | %CER 9.87 [32629 / 330498, 2039 ins, 11994 del, 18596 sub ]
aishell-test    %SER 46.71 | %CER 6.63 [6951 / 104765, 511 ins, 114 del, 6326 sub ]
test_meeting    %SER 90.96 | %CER 19.41 [42785 / 220385, 2305 ins, 17977 del, 22503 sub ]
test_net        %SER 63.34 | %CER 12.16 [50563 / 415746, 3467 ins, 9473 del, 37623 sub ]

0.04 0.9
dev     %SER 69.08 | %CER 10.35 [34223 / 330498, 1665 ins, 13278 del, 19280 sub ]

alpga 0.05 beta 0.8
dev     %SER 69.22 | %CER 10.41 [34402 / 330498, 1615 ins, 13566 del, 19221 sub ]

alpha 0.05 beta 0.7
dev     %SER 69.35 | %CER 10.47 [34590 / 330498, 1569 ins, 13838 del, 19183 sub ]

alpha 0.1 beta 0.7
dev     %SER 69.13 | %CER 10.53 [34787 / 330498, 1497 ins, 14576 del, 18714 sub ]

alpha 0.05
dev     %SER 69.53 | %CER 10.55 [34882 / 330498, 1524 ins, 14230 del, 19128 sub ]

alpha 0.1
aishell-test    %SER 47.16 | %CER 6.71 [7032 / 104765, 261 ins, 174 del, 6597 sub ]
test_meeting    %SER 90.97 | %CER 20.21 [44533 / 220385, 1415 ins, 21368 del, 21750 sub ]
test_net        %SER 64.17 | %CER 12.40 [51549 / 415746, 2150 ins, 12248 del, 37151 sub ]
dev     %SER 69.29 | %CER 10.61 [35067 / 330498, 1450 ins, 14965 del, 18652 sub ]

alpha 0.15
dev     %SER 69.14 | %CER 10.70 [35366 / 330498, 1398 ins, 15736 del, 18232 sub ]
test_meeting    %SER 91.10 | %CER 20.44 [45045 / 220385, 1327 ins, 22379 del, 21339 sub ]
test_net        %SER 63.91 | %CER 12.37 [51448 / 415746, 2038 ins, 12851 del, 36559 sub ]
aishell-test    %SER 46.14 | %CER 6.54 [6853 / 104765, 239 ins, 199 del, 6415 sub ]

alpha 0.2
dev     %SER 69.32 | %CER 10.86 [35896 / 330498, 1349 ins, 16598 del, 17949 sub ]
test_meeting    %SER 91.24 | %CER 20.74 [45716 / 220385, 1242 ins, 23454 del, 21020 sub ]
test_net        %SER 63.87 | %CER 12.40 [51542 / 415746, 1936 ins, 13467 del, 36139 sub ]
aishell-test    %SER 45.68 | %CER 6.43 [6732 / 104765, 221 ins, 226 del, 6285 sub ]
```

### Tuning on aishell-1

```
no lm
aishell-test    %SER 49.99 | %CER 7.22 [7560 / 104765, 249 ins, 201 del, 7110 sub ] 

alpha 0.15
aishell-test    %SER 44.58 | %CER 6.32 [6618 / 104765, 226 ins, 185 del, 6207 sub ]

alpha 0.2
aishell-test    %SER 43.48 | %CER 6.17 [6463 / 104765, 210 ins, 207 del, 6046 sub ]

alpha 0.25
aishell-test    %SER 42.89 | %CER 6.06 [6350 / 104765, 196 ins, 215 del, 5939 sub ]

alpha 0.3
aishell-test    %SER 42.60 | %CER 6.01 [6298 / 104765, 181 ins, 229 del, 5888 sub ]

alpha 0.35
aishell-test    %SER 42.15 | %CER 5.93 [6209 / 104765, 175 ins, 248 del, 5786 sub ]

alpha 0.4
aishell-test    %SER 41.99 | %CER 5.91 [6192 / 104765, 162 ins, 270 del, 5760 sub ]

alpha 0.45
aishell-test    %SER 41.65 | %CER 5.88 [6165 / 104765, 157 ins, 282 del, 5726 sub ]

alpha 0.5
aishell-test    %SER 41.64 | %CER 5.89 [6167 / 104765, 153 ins, 301 del, 5713 sub ]

alpha 0.55
aishell-test    %SER 41.72 | %CER 5.89 [6174 / 104765, 148 ins, 316 del, 5710 sub ]

alpha 0.6
aishell-test    %SER 41.81 | %CER 5.91 [6195 / 104765, 148 ins, 333 del, 5714 sub ]

alpha 0.7
aishell-test    %SER 41.95 | %CER 5.95 [6230 / 104765, 148 ins, 353 del, 5729 sub ]
```

### Monitor figure
![monitor](./monitor.png)
