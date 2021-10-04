### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 11.92
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* `v1` with smaller peak factor and less warmup steps
* As the results shown, the model seems overfitting at very early stage.

### WER
```
Use CPU = False
test %WER 13.97 [9003 / 64428, 1066 ins, 766 del, 7171 sub ]
     %CER 7.26 [7609 / 104765, 180 ins, 224 del, 7205 sub ]

Custom checkpoint: avg_best_10.pt
Use CPU = False
test %WER 12.99 [8366 / 64428, 1008 ins, 707 del, 6651 sub ]
     %CER 6.58 [6892 / 104765, 153 ins, 190 del, 6549 sub ]

Custom checkpoint: avg_50_10.pt
Use CPU = False
test %WER 13.05 [8406 / 64428, 1004 ins, 733 del, 6669 sub ]
     %CER 6.53 [6836 / 104765, 124 ins, 244 del, 6468 sub ]
```

### Monitor figure
![monitor](./monitor.png)
