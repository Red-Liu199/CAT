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
dev             %SER 71.07 | %CER 11.16 [36887 / 330498, 1279 ins, 16227 del, 19381 sub ]
test_meeting    %SER 91.77 | %CER 20.97 [46214 / 220385, 1224 ins, 22925 del, 22065 sub ]
test_net        %SER 65.49 | %CER 12.76 [53035 / 415746, 1943 ins, 12961 del, 38131 sub ]
aishell-test    %SER 49.99 | %CER 7.22 [7560 / 104765, 249 ins, 201 del, 7110 sub ] 

+lm-aishell
%SER 41.23 | %CER 5.83 [ 6104 / 104765, 200 ins, 218 del, 5686 sub ]    alpha = 0.5 | beta = 1.5

+lm-aishell-word
%SER 40.48 | %CER 5.70 [ 5976 / 104765, 165 ins, 306 del, 5505 sub ]    alpha = 0.53125 | beta = 1.0

+lm-trans-word
%SER 43.21 | %CER 6.15 [ 6443 / 104765, 169 ins, 381 del, 5893 sub ]    alpha = 0.5 | beta = 0.0

+lm-trans
%SER 67.12 | %CER 9.81 [ 32416 / 330498, 2392 ins, 11206 del, 18818 sub ]       alpha = 0.28125 | beta = 3.5
%SER 63.08 | %CER 12.12 [ 50404 / 415746, 3290 ins, 9963 del, 37151 sub ]       alpha = 0.28125 | beta = 2.5
%SER 90.76 | %CER 19.39 [ 42731 / 220385, 2139 ins, 18343 del, 22249 sub ]      alpha = 0.21875 | beta = 2.25
%SER 44.09 | %CER 6.22 [ 6516 / 104765, 222 ins, 250 del, 6044 sub ]            alpha = 0.50 | beta = 1.50

+lm-ext-5mil
%SER 66.24 | %CER 9.66 [ 31918 / 330498, 2540 ins, 11140 del, 18238 sub ]       alpha = 0.375 | beta = 4.25
%SER 62.23 | %CER 11.90 [ 49493 / 415746, 3042 ins, 10297 del, 36154 sub ]      alpha = 0.28125 | beta = 2.25
%SER 90.50 | %CER 19.23 [ 42376 / 220385, 2246 ins, 18196 del, 21934 sub ]      alpha = 0.28125 | beta = 2.75

+lm-ext-10mil
%SER 66.13 | %CER 9.63 [ 31841 / 330498, 2453 ins, 11065 del, 18323 sub ]       alpha = 0.28125 | beta = 3.75
%SER 61.88 | %CER 11.83 [ 49177 / 415746, 3022 ins, 10316 del, 35839 sub ]      alpha = 0.28125 | beta = 2.25
%SER 90.30 | %CER 19.23 [ 42390 / 220385, 2040 ins, 18470 del, 21880 sub ]      alpha = 0.1875 | beta = 2.0

+lm-ext-15mil
%SER 66.08 | %CER 9.62 [ 31797 / 330498, 2367 ins, 11243 del, 18187 sub ]       alpha = 0.28125 | beta = 3.5
%SER 61.80 | %CER 11.82 [ 49148 / 415746, 2906 ins, 10459 del, 35783 sub ]      alpha = 0.25 | beta = 2.0
%SER 90.31 | %CER 19.18 [ 42259 / 220385, 2205 ins, 18239 del, 21815 sub ]      alpha = 0.28125 | beta = 2.75
```


### Monitor figure
![monitor](./monitor.png)
