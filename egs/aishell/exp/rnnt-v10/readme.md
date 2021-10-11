### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 90.33
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* `v5` + rm spaces in transcript

### WER
```
Custom checkpoint: avg_best_10.pt
Use CPU = True
test ext_lm= %CER 5.21 [5460 / 104765, 157 ins, 137 del, 5166 sub ]

Use CPU = False
test ext_lm= %CER 5.91 [6194 / 104765, 145 ins, 200 del, 5849 sub ]

Custom checkpoint: avg_best_10.pt
Use CPU = False
test ext_lm= %CER 5.21 [5460 / 104765, 157 ins, 137 del, 5166 sub ]

Custom checkpoint: avg_last_10.pt
Use CPU = False
test ext_lm= %CER 5.41 [5672 / 104765, 96 ins, 243 del, 5333 sub ]
```

### Monitor figure
![monitor](./monitor.png)
