### Result

```
PPL over libri test set is: 74.99

PPL over TLv2 test set is: 27.58
```

### LM adaptation

- with `rnnt-v27`

```
baseline
tlv2-dev	%SER 86.39 | %WER 11.67 [2075 / 17783, 521 ins, 297 del, 1257 sub ]
tlv2-test	%SER 77.06 | %WER 11.40 [3135 / 27500, 548 ins, 581 del, 2006 sub ]

+lm-v11 libri trans
tlv2-dev	%SER 86.39 | %WER 11.48 [2041 / 17783, 515 ins, 288 del, 1238 sub ]
tlv2-test	%SER 76.97 | %WER 11.16 [3069 / 27500, 544 ins, 566 del, 1959 sub ]

+lm-tlv2 tlv2 trans
tlv2-dev	%SER 83.83 | %WER 10.82 [1924 / 17783, 469 ins, 313 del, 1142 sub ]
tlv2-test	%SER 74.29 | %WER 10.24 [2816 / 27500, 453 ins, 624 del, 1739 sub ]
```

