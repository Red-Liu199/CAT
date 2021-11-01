### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 120.42
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* follow google 2048 batch size and large conformer

### WER
```
Custom checkpoint: avg_last_10.pt | Use CPU = True
test_clean %WER 2.37 [1245 / 52576, 139 ins, 99 del, 1007 sub ]
    oracle %WER 2.00 [1050 / 52576, 121 ins, 76 del, 853 sub ]
test_other %WER 5.46 [2857 / 52343, 316 ins, 234 del, 2307 sub ]
    oracle %WER 4.63 [2423 / 52343, 251 ins, 193 del, 1979 sub ]

Custom checkpoint: avg_best_10.pt | Use CPU = True
test_clean %WER 2.41 [1268 / 52576, 142 ins, 104 del, 1022 sub ]
    oracle %WER 1.96 [1030 / 52576, 121 ins, 73 del, 836 sub ]
test_other %WER 5.44 [2850 / 52343, 314 ins, 239 del, 2297 sub ]
    oracle %WER 4.54 [2378 / 52343, 241 ins, 192 del, 1945 sub ]
```

### Monitor figure
![monitor](./monitor.png)
