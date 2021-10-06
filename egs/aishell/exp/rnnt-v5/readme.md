### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 90.33
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* `v3` + char modeling, gradient clipping and attention dropout

### WER
```
Use CPU = False
test %WER 13.02 [8390 / 64428, 1100 ins, 705 del, 6585 sub ]
%CER 6.66 [6976 / 104765, 158 ins, 159 del, 6659 sub ]

Custom checkpoint: avg_best_10.pt
Use CPU = False
test %WER 12.27 [7905 / 64428, 1030 ins, 664 del, 6211 sub ]
%CER 6.17 [6463 / 104765, 128 ins, 145 del, 6190 sub ]

Custom checkpoint: avg_last_10.pt
Use CPU = False
test %WER 12.24 [7888 / 64428, 1043 ins, 649 del, 6196 sub ]
%CER 6.16 [6457 / 104765, 122 ins, 151 del, 6184 sub ]
```

### Monitor figure
![monitor](./monitor.png)
