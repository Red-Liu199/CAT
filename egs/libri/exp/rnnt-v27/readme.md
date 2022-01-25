### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 120.42
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* follow google 2048 batch size and large conformer

### Result
```
baseline ALSD 16
test-clean/test-other   2.38/5.40

RNA 16
dev_clean       %SER 27.64 | %WER 2.18 [1185 / 54402, 126 ins, 100 del, 959 sub ]
dev_other       %SER 45.32 | %WER 5.34 [2720 / 50948, 251 ins, 218 del, 2251 sub ]
test_clean      %SER 28.55 | %WER 2.39 [1259 / 52576, 145 ins, 101 del, 1013 sub ]
test_other      %SER 46.48 | %WER 5.41 [2830 / 52343, 302 ins, 238 del, 2290 sub ]
+lm-v10 rescored with alpha=0.75 beta=0.875
dev_clean       %SER 25.12 | %WER 1.90 [1034 / 54402, 109 ins, 102 del, 823 sub ]
dev_other       %SER 37.78 | %WER 4.34 [2212 / 50948, 206 ins, 191 del, 1815 sub ]
test_clean      %SER 25.27 | %WER 2.05 [1079 / 52576, 126 ins, 108 del, 845 sub ]
test_other      %SER 40.56 | %WER 4.55 [2383 / 52343, 250 ins, 243 del, 1890 sub ]
```

### Monitor figure
![monitor](./monitor.png)
