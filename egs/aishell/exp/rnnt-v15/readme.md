### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 84.30
* GPU info \[9\]
  * \[9\] NVIDIA GeForce RTX 3090

### Appendix

* `v14` + bos token

### Result
```
baseline
test    %CER 4.82 [5051 / 104765, 157 ins, 131 del, 4763 sub ]
test    %CER 1.63 [1705 / 104765, 58 ins, 69 del, 1578 sub ]
rna topo
test    %CER 4.79 [5023 / 104765, 134 ins, 131 del, 4758 sub ]
test    %CER 2.27 [2378 / 104765, 69 ins, 88 del, 2221 sub ]

fusion with 0.5 lm-v4
test    %CER 3.67

fusion with 0.15 lm-v5 
test    %CER 4.69
rna topo decode
test    %CER 4.68 [4908 / 104765, 124 ins, 146 del, 4638 sub ]
```

### Monitor figure
![monitor](./monitor.png)
