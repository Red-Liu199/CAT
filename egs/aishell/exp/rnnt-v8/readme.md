### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 90.33
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* `v5` + remove gradient clipping + attention dropout 0.3 -> 0.2
* or `v7` + attention dropout 0.2 -> 0.1

### WER
```
test    %CER 5.46 [5718 / 104765, 118 ins, 120 del, 5480 sub ]
test    %CER 3.81 [3991 / 104765, 82 ins, 106 del, 3803 sub ]
```

### Monitor figure
![monitor](./monitor.png)
