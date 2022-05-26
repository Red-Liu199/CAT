
train command:

```bash
utils/pipeline/ngram.sh exp/template-lm-ngram --text-corpus -o 3 --type probing
```

property:

- prune: 
- type:  probing
- size:  17MB

perplexity:

```
Test file: data/local-lm/ptb.valid.txt -> ppl: 252.96
Test file: data/local-lm/ptb.test.txt -> ppl: 268.65
```
