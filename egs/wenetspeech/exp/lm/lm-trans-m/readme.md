
train command:

```bash
utils/pipeline/ngram.sh exp/lm/lm-trans-m
```

property:

- prune: 
- type:  trie
- size:  207MB

perplexity:

```
Test file: dev.tmp -> ppl: 66.74
Test file: test_net.tmp -> ppl: 75.89
Test file: test_meeting.tmp -> ppl: 65.65
```
