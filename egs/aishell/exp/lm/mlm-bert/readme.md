## Fine-tune BERT
We use the [LM pipeline](../../README.md) to fine-tune BERT on aishell.
```
python utils/pipeline/lm.py exp/lm/mlm-bert --ngpu 4
```
The pipeline includes 4 stages:
```
(data prepare) ->
tokenizer training -> data packing -> nn training -> inference
```

### Notes

* In **stage 2 (data packing)**, if you use a `PretrainedTokenizer` of type `BertTokenizer` to tokenize the data, the start token *[CLS]* and end token *[SEP]* will be added at the beginning and end of each sentence automatically. This is incompatible with the pipeline since the pipeline will automatically add another start token *0* at the beginning. So we need to delete the duplicated start token after packing data
```
python utils/reprocess.py exp/mlm-bert/lmbin exp/mlm-bert/lmbin --head_del 1
```
* Then, prepare masked training samples after packing data.
```
python utils/reprocess.py exp/mlm-bert/lmbin exp/mlm-bert/lmbin --mlm
```
### Result
|CER type     | BERT |  BERT after fine-tuning  |
| -------     | -------- | ----------- |
| in-domain   | 3.29     |  3.11       | 
| cross-domain| 3.65     |  3.44       | 


|     training process    |
|:-----------------------:|
|![monitor](./monitor.png)|
