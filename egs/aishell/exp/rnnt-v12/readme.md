### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 90.33
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* `v10` + prediction network mask, stop epochs: 100 -> 80

### WER
```
Use CPU = False
test ext_lm= %CER 5.48 [5743 / 104765, 148 ins, 190 del, 5405 sub ]

Custom checkpoint: avg_best_10.pt
Use CPU = False
test ext_lm= %CER 4.87 [5102 / 104765, 137 ins, 141 del, 4824 sub ]

Custom checkpoint: avg_last_10.pt
Use CPU = False
test ext_lm= %CER 4.85 [5078 / 104765, 112 ins, 163 del, 4803 sub ]
```

### Monitor figure
![monitor](./monitor.png)
