
train command:

```bash
utils/pipeline/ngram.sh exp/lm/lm-trans-l --text-corpus
```

property:

- prune: 
- type:  trie
- size:  1.3GB

perplexity:

```
Test file: aishell-test.tmp -> ppl: 68.90
Test file: test_net.tmp -> ppl: 59.07
Test file: test_meeting.tmp -> ppl: 55.39
```
