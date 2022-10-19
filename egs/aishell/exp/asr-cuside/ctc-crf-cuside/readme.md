### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 117.07
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Notes

* CTC-CRF-CUSIDE experiment,training 25 epochs 
* In decoding, a word-level 3-gram language model (LM) trained on the transcripts is used to build the decoding WFST.

### Result
```
%CER 5.98 [ 1255 / 20986, 21 ins, 42 del, 1192 sub ] [PARTIAL] exp/catv3/ctc-crf-last5-40-80-40/cer_11_0.0
```

|     training process    |
|:-----------------------:|
|![monitor](./monitor.png)|
