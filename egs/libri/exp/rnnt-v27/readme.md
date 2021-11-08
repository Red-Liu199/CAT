### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 120.42
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* follow google 2048 batch size and large conformer

### Result
```
lc
test_clean      %WER 2.37 [1245 / 52576, 144 ins, 98 del, 1003 sub ]
test_clean      %WER 1.98 [1039 / 52576, 124 ins, 76 del, 839 sub ]
test_other      %WER 5.44 [2850 / 52343, 304 ins, 236 del, 2310 sub ]
test_other      %WER 4.56 [2385 / 52343, 240 ins, 189 del, 1956 sub ]

native:
test_clean      %WER 2.38 [1252 / 52576, 145 ins, 98 del, 1009 sub ]
test_clean      %WER 1.23 [648 / 52576, 64 ins, 43 del, 541 sub ]
test_other      %WER 5.44 [2850 / 52343, 307 ins, 236 del, 2307 sub ]
test_other      %WER 3.44 [1800 / 52343, 179 ins, 118 del, 1503 sub ]
```

### Monitor figure
![monitor](./monitor.png)
