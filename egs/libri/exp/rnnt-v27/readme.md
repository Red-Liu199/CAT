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

fusion with lm-v9 0.45, lm trained 28 epochs
test-clean/test-other   1.96/4.44

fusion with lm-v9 0.45, lm trained 35 epochs
test_clean      %WER 1.95 [1025 / 52576, 128 ins, 98 del, 799 sub ]
test_other      %WER 4.46 [2334 / 52343, 235 ins, 281 del, 1818 sub ]

```

### Monitor figure
![monitor](./monitor.png)
