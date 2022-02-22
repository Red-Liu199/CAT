### Result

```
PPL over libri test set is: 74.99

PPL over TLv2 test set is: 27.58
```

### LM adaptation

* RNNT: `rnnt-v27`
* CRF: `crf-v1`

```
PPL
Test set: exp/lm-tlv2//lmbin/test-data_tlv2_dev_txt.pkl -> ppl: 27.67
Test set: exp/lm-tlv2//lmbin/test-data_tlv2_test_txt.pkl -> ppl: 27.58

RNNT+lm-tlv2 tlv2 trans tuned
tlv2-dev	%SER 84.62 | %WER 10.82 [ 1925 / 17783, 463 ins, 318 del, 1144 sub ]    0.5     1.0
tlv2-test	%SER 74.11 | %WER 10.23 [ 2813 / 27500, 453 ins, 624 del, 1736 sub ]    0.5625  1.5

CRF + libri corpus LM
tlv2-dev	%WER 12.24 [ 2176 / 17783, 529 ins, 431 del, 1216 sub ] exp/crf-v1//fglarge/tlv2-dev/wer_13_1.0
tlv2-test	%WER 11.86 [ 3262 / 27500, 558 ins, 753 del, 1951 sub ] exp/crf-v1//fglarge/tlv2-test/wer_13_1.0

CRF + tlv2 corpus LM
tlv2-dev	%WER 10.38 [ 1845 / 17783, 453 ins, 360 del, 1032 sub ] exp/crf-v1/tlv2-fg/tlv2-dev/wer_17_0.5
tlv2-test	%WER 9.81 [ 2697 / 27500, 479 ins, 624 del, 1594 sub ] exp/crf-v1/tlv2-fg/tlv2-test/wer_16_0.5
```

