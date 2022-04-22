### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 84.30
* GPU info \[9\]
  * \[9\] GeForce RTX 3090

### Appendix

* `v12` + subsample: conv2d -> vgg2l, stop epochs: 80 -> 100

### Result
```
lc beam 10
test    %CER 4.77 [4998 / 104765, 140 ins, 132 del, 4726 sub ]
test    %CER 3.41 [3575 / 104765, 79 ins, 116 del, 3380 sub ]

alsd beam 64
Time of searching: 808.10s
test    %CER 4.76 [4990 / 104765, 130 ins, 134 del, 4726 sub ]
test    %CER 1.68 [1759 / 104765, 43 ins, 74 del, 1642 sub ]
```

### Monitor figure
![monitor](./monitor.png)
