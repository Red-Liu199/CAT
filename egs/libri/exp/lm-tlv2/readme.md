### Result

```
PPL over libri test set is: 74.99

PPL over TLv2 test set is: 27.58
```

### LM adaptation

* RNNT: `rnnt-v27`
* CRF: `crf-v1`

```
RNNT baseline
tlv2-dev	%SER 86.39 | %WER 11.67 [2075 / 17783, 521 ins, 297 del, 1257 sub ]
tlv2-test	%SER 77.06 | %WER 11.40 [3135 / 27500, 548 ins, 581 del, 2006 sub ]

RNNT+lm-v11 libri trans
tlv2-dev	%SER 86.39 | %WER 11.48 [2041 / 17783, 515 ins, 288 del, 1238 sub ]
tlv2-test	%SER 76.97 | %WER 11.16 [3069 / 27500, 544 ins, 566 del, 1959 sub ]

RNNT+lm-tlv2 tlv2 trans tuned
tlv2-dev	%SER 84.62 | %WER 10.82 [ 1925 / 17783, 463 ins, 318 del, 1144 sub ]    0.5     1.0
tlv2-test	%SER 74.11 | %WER 10.23 [ 2813 / 27500, 453 ins, 624 del, 1736 sub ]    0.5625  1.5

CRF + libri corpus LM
tlv2-dev	%WER 12.24 [ 2176 / 17783, 529 ins, 431 del, 1216 sub ] exp/crf-v1//fglarge/tlv2-dev/wer_13_1.0
tlv2-test	%WER 11.86 [ 3262 / 27500, 558 ins, 753 del, 1951 sub ] exp/crf-v1//fglarge/tlv2-test/wer_13_1.0

CRF + tlv2 corpus LM
tlv2-dev	%WER 29.25 [ 5201 / 17783, 630 ins, 896 del, 3675 sub ] exp/crf-v1/tlv2-tgprune/tlv2-dev/wer_17_0.0
tlv2-test	%WER 26.74 [ 7354 / 27500, 740 ins, 1327 del, 5287 sub ] exp/crf-v1/tlv2-tgprune/tlv2-test/wer_17_0.0
```

