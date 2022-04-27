### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 84.30
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* ported from rnnt-v15
* use torchaudio for feature extraction w/o CMVN

### Result

compared to baseline `rnnt-v15`

| model               | dev  | test |
| ------------------- | ---- | ---- |
| kaldi prep w/ CMVN  | 4.44 | 4.80 |
| torchaudio w/o CMVN | 4.41 | 4.87 | 

```
dev     %SER 33.99 | %CER 4.41 [ 9059 / 205341, 249 ins, 160 del, 8650 sub ]
test    %SER 36.41 | %CER 4.87 [ 5099 / 104765, 139 ins, 139 del, 4821 sub ]
```

### Monitor figure
![monitor](./monitor.png)
