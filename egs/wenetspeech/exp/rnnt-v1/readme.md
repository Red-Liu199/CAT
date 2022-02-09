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

+lm-trans
alpha=0.1875 beta=3.0625
dev             %SER 67.14 | %CER 9.83 [32503 / 330498, 2376 ins, 11122 del, 19005 sub ]
test_meeting    %SER 91.10 | %CER 19.44 [42851 / 220385, 2594 ins, 17469 del, 22788 sub ]
test_net        %SER 63.85 | %CER 12.26 [50990 / 415746, 4181 ins, 8878 del, 37931 sub ]
aishell-test    %SER 47.66 | %CER 6.85 [7180 / 104765, 699 ins, 97 del, 6384 sub ]

+lm-ext-5mil
alpha=0.25 beta=3.53125
dev             %SER 66.33 | %CER 9.68 [31987 / 330498, 2450 ins, 11079 del, 18458 sub ]
test_meeting    %SER 90.72 | %CER 19.30 [42542 / 220385, 2663 ins, 17452 del, 22427 sub ]
test_net        %SER 63.05 | %CER 12.06 [50159 / 415746, 4410 ins, 8746 del, 37003 sub ]
aishell-test    %SER 45.37 | %CER 6.51 [6821 / 104765, 732 ins, 100 del, 5989 sub ]

+lm-ext-10mil
alpha=0.25 beta=3.6875
dev             %SER 66.18 | %CER 9.65 [31894 / 330498, 2483 ins, 10994 del, 18417 sub ]
test_meeting    %SER 90.66 | %CER 19.30 [42535 / 220385, 2730 ins, 17408 del, 22397 sub ]
test_net        %SER 62.87 | %CER 12.03 [50022 / 415746, 4538 ins, 8653 del, 36831 sub ]
aishell-test    %SER 45.30 | %CER 6.50 [6806 / 104765, 767 ins, 98 del, 5941 sub ]

+lm-ext-15mil
alpha=0.25 beta=3.6875
dev             %SER 66.11 | %CER 9.63 [31820 / 330498, 2374 ins, 11197 del, 18249 sub ]
test_meeting    %SER 90.48 | %CER 19.23 [42385 / 220385, 2570 ins, 17585 del, 22230 sub ]
test_net        %SER 62.27 | %CER 11.94 [49642 / 415746, 4188 ins, 8886 del, 36568 sub ]
aishell-test    %SER 43.99 | %CER 6.33 [6631 / 104765, 669 ins, 103 del, 5859 sub ]

+lm trans ? need to be updated
dev             %SER 67.00 | %CER 9.80 [32384 / 330498, 2439 ins, 11058 del, 18887 sub ]  alpha = 0.25 | beta = 3.53125
test_meeting    %SER 90.85 | %CER 19.40 [42749 / 220385, 2210 ins, 18117 del, 22422 sub ]  alpha = 0.19 | beta = 2.28
test_net        %SER 62.97 | %CER 12.11 [50361 / 415746, 2923 ins, 10528 del, 36910 sub ]  alpha = 0.31 | beta = 2.28
aishell-test    %SER 44.15 | %CER 6.23 [6525 / 104765, 220 ins, 256 del, 6049 sub ]  alpha = 0.50 | beta = 1.50
```


### Monitor figure
![monitor](./monitor.png)
