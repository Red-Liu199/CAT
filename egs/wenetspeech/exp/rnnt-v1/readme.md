### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 91.27
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* trained on wenet speech M subset (1000 hour speech)

### Result
```
dev     %SER 71.07 | %CER 11.16 [36887 / 330498, 1279 ins, 16227 del, 19381 sub ]
dev     %SER 51.67 | %CER 7.38 [24397 / 330498, 736 ins, 11046 del, 12615 sub ]

test_meeting  %SER 91.77 | %CER 20.97 [46214 / 220385, 1224 ins, 22925 del, 22065 sub ]
test_meeting  %SER 82.62 | %CER 16.22 [35757 / 220385, 737 ins, 18211 del, 16809 sub ]

test_net      %SER 65.49 | %CER 12.76 [53035 / 415746, 1943 ins, 12961 del, 38131 sub ]
test_net      %SER 43.43 | %CER 8.23 [34216 / 415746, 1083 ins, 9006 del, 24127 sub ]
```

### Tuning

```
no lm
aishell-test    %SER 49.99 | %CER 7.22 [7560 / 104765, 249 ins, 201 del, 7110 sub ] 

alpha 0.15 beta 0.6
aishell-test    %SER 50.33 | %CER 7.29 [7642 / 104765, 196 ins, 291 del, 7155 sub ]

```

### Monitor figure
![monitor](./monitor.png)
