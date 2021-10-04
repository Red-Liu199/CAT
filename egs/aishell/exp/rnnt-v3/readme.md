### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 11.92
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* settings refers to [ESPNET aishell](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md#conformer-transducer)

### WER
```
Use CPU = False
test %WER 13.10 [8438 / 64428, 950 ins, 809 del, 6679 sub ]
     %CER 6.20 [6492 / 104765, 137 ins, 176 del, 6179 sub ]

Custom checkpoint: avg_best_10.pt
Use CPU = False
test %WER 11.43 [7361 / 64428, 748 ins, 780 del, 5833 sub ]
     %CER 5.31 [5559 / 104765, 141 ins, 124 del, 5294 sub ]

Custom checkpoint: avg_last_10.pt
Use CPU = False
test %WER 11.61 [7479 / 64428, 790 ins, 756 del, 5933 sub ]
     %CER 5.42 [5678 / 104765, 116 ins, 160 del, 5402 sub ]
```

### Monitor figure
![monitor](./monitor.png)
