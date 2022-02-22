### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 120.42
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* follow google 2048 batch size and large conformer

### Result
```
baseline ALSD 16
test-clean/test-other   2.38/5.40

RNA 16
dev_clean       %SER 27.64 | %WER 2.18 [1185 / 54402, 126 ins, 100 del, 959 sub ]
dev_other       %SER 45.32 | %WER 5.34 [2720 / 50948, 251 ins, 218 del, 2251 sub ]
test_clean      %SER 28.55 | %WER 2.39 [1259 / 52576, 145 ins, 101 del, 1013 sub ]
test_other      %SER 46.48 | %WER 5.41 [2830 / 52343, 302 ins, 238 del, 2290 sub ]
tlv2-dev        %SER 86.39 | %WER 11.67 [ 2075 / 17783, 521 ins, 297 del, 1257 sub ]
tlv2-test       %SER 77.06 | %WER 11.40 [ 3135 / 27500, 548 ins, 581 del, 2006 sub ]

RNA 128
dev_clean       %SER 27.64 | %WER 2.18 [ 1188 / 54402, 126 ins, 100 del, 962 sub ]
dev_other       %SER 45.32 | %WER 5.33 [ 2716 / 50948, 250 ins, 217 del, 2249 sub ]
test_clean      %SER 28.51 | %WER 2.40 [ 1262 / 52576, 145 ins, 103 del, 1014 sub ]
test_other      %SER 46.51 | %WER 5.42 [ 2836 / 52343, 302 ins, 236 del, 2298 sub ]
tlv2-dev        %SER 86.39 | %WER 11.67 [ 2075 / 17783, 519 ins, 297 del, 1259 sub ]
tlv2-test       %SER 77.06 | %WER 11.41 [ 3137 / 27500, 550 ins, 584 del, 2003 sub ]

fusion + lm-v10 beam 16

test_clean      %SER 24.31 | %WER 1.95 [ 1023 / 52576, 119 ins, 99 del, 805 sub ]     [0.53125, 0.75]

+lm-v13 tuned
%SER 23.94 | %WER 1.81 [ 983 / 54402, 102 ins, 90 del, 791 sub ]        [0.5625, 1.0]
%SER 36.56 | %WER 4.03 [ 2052 / 50948, 169 ins, 217 del, 1666 sub ]     [0.9375, 1.0]
%SER 24.31 | %WER 1.94 [ 1022 / 52576, 116 ins, 103 del, 803 sub ]      [0.5625, 0.75]
%SER 39.84 | %WER 4.39 [ 2298 / 52343, 229 ins, 255 del, 1814 sub ]     [0.5, 0.5]
%SER 82.45 | %WER 10.76 [ 1914 / 17783, 462 ins, 312 del, 1140 sub ]    [0.6875, 1.5]
%SER 74.37 | %WER 10.43 [ 2869 / 27500, 464 ins, 668 del, 1737 sub ]    [0.5, 1.0]

+lm-v12 tuned
%SER 25.86 | %WER 2.03 [ 1105 / 54402, 120 ins, 106 del, 879 sub ]      [0.34375, 0.5]
%SER 40.08 | %WER 4.62 [ 2352 / 50948, 199 ins, 236 del, 1917 sub ]     [0.53125, 0.75]
%SER 26.30 | %WER 2.15 [ 1129 / 52576, 127 ins, 110 del, 892 sub ]      [0.40625, 0.75]
%SER 43.04 | %WER 4.82 [ 2521 / 52343, 251 ins, 272 del, 1998 sub ]     [0.375, 0.5]
%SER 85.21 | %WER 11.09 [ 1973 / 17783, 497 ins, 294 del, 1182 sub ]    [0.25, 0.75]
%SER 75.84 | %WER 10.78 [ 2964 / 27500, 508 ins, 608 del, 1848 sub ]    [0.25, 0.75]

+lm-tlv2 tuned
%SER 27.60 | %WER 2.18 [ 1184 / 54402, 119 ins, 102 del, 963 sub ]      [0.03125, 0.25]
%SER 45.11 | %WER 5.31 [ 2707 / 50948, 245 ins, 230 del, 2232 sub ]     [0.125, 0.25]
%SER 28.02 | %WER 2.36 [ 1241 / 52576, 141 ins, 108 del, 992 sub ]      [0.09375, 0.25]
%SER 46.44 | %WER 5.39 [ 2820 / 52343, 296 ins, 243 del, 2281 sub ]     [0.03125, 0.0]
%SER 82.25 | %WER 10.25 [ 1823 / 17783, 428 ins, 331 del, 1064 sub ]    [0.5625, 1.25]
%SER 73.51 | %WER 9.96 [ 2740 / 27500, 427 ins, 645 del, 1668 sub ]     [0.53125, 1.5]
```

### Density ratio

```
density ratio lm-v14 lm-v13 beta=0.2
%SER 23.31 | %WER 1.76 [ 960 / 54402, 104 ins, 88 del, 768 sub ]        [-0.1875, 0.625]
%SER 36.28 | %WER 3.98 [ 2026 / 50948, 166 ins, 210 del, 1650 sub ]     [-0.25, 0.9375]
%SER 23.85 | %WER 1.90 [ 1000 / 52576, 115 ins, 95 del, 790 sub ]       [-0.25, 0.625]
%SER 39.03 | %WER 4.26 [ 2232 / 52343, 209 ins, 271 del, 1752 sub ]     [-0.25, 0.8125]
%SER 82.05 | %WER 10.62 [ 1888 / 17783, 459 ins, 320 del, 1109 sub ]    [-0.25, 0.625]
%SER 74.37 | %WER 10.32 [ 2838 / 27500, 468 ins, 637 del, 1733 sub ]    [-0.25, 0.5]

density ratio lm-v14 lm-tlv2 beta 0.7
%SER 27.27 | %WER 2.13 [ 1158 / 54402, 114 ins, 101 del, 943 sub ]      [0.1875, 0.0]
%SER 43.40 | %WER 5.11 [ 2603 / 50948, 234 ins, 212 del, 2157 sub ]     [0.25, 0.0]
%SER 27.98 | %WER 2.36 [ 1242 / 52576, 136 ins, 104 del, 1002 sub ]     [0.125, 0.09375]
%SER 45.49 | %WER 5.26 [ 2754 / 52343, 286 ins, 270 del, 2198 sub ]     [0.1875, 0.09375]
%SER 83.43 | %WER 10.33 [ 1837 / 17783, 423 ins, 340 del, 1074 sub ]    [-0.0625, 0.53125]
%SER 72.81 | %WER 9.72 [ 2674 / 27500, 417 ins, 640 del, 1617 sub ]     [-0.25, 0.59375]
```

### Monitor figure
![monitor](./monitor.png)
