### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 90.33
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* `v5` + remove gradient clipping + attention dropout 0.3 -> 0.2
* gradient clipping affects little accordings a few epochs of running of `v6`

### WER
```
Use CPU = False
test %WER 12.41 [7994 / 64428, 1076 ins, 629 del, 6289 sub ]
%CER 6.24 [6533 / 104765, 151 ins, 131 del, 6251 sub ]

Custom checkpoint: avg_best_10.pt
Use CPU = False
test %WER 11.59 [7464 / 64428, 954 ins, 633 del, 5877 sub ]
%CER 5.67 [5936 / 104765, 114 ins, 127 del, 5695 sub ]

Custom checkpoint: avg_last_10.pt
Use CPU = False
test %WER 11.69 [7533 / 64428, 1002 ins, 615 del, 5916 sub ]
%CER 5.68 [5955 / 104765, 117 ins, 125 del, 5713 sub ]
```

### Monitor figure
![monitor](./monitor.png)
