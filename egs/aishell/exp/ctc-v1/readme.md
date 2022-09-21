### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 79.50
* GPU info \[8\]
  * \[8\] NVIDIA GeForce RTX 3090

### Notes

* CTC experiment, using the same encoder as `rnnt/rnnt-v15`

### Result
```
dev     %SER 39.52 | %CER 5.13 [ 10537 / 205341, 184 ins, 174 del, 10179 sub ]
test    %SER 42.13 | %CER 5.78 [ 6058 / 104765, 115 ins, 135 del, 5808 sub ]

+lm-v5 trans char 5-gram a=0.2
dev     %SER 37.45 | %CER 4.89 [ 10044 / 205341, 161 ins, 191 del, 9692 sub ]
test    %SER 39.09 | %CER 5.39 [ 5645 / 104765, 82 ins, 156 del, 5407 sub ]

+lm-v6 trans word 3-gram a=0.28 b=-0.5
dev     %SER 34.93 | %CER 4.63 [ 9498 / 205341, 159 ins, 203 del, 9136 sub ]
test    %SER 36.51 | %CER 5.08 [ 5327 / 104765, 76 ins, 162 del, 5089 sub ]
```

|     training process    |
|:-----------------------:|
|![monitor](./monitor.png)|
