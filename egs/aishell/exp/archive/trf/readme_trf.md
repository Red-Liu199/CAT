
## Liu Haochen's addition

The environmental configuration process is the same as that of Huahuan. Refer to the instructions of Huahuan, additional items need to be added (discrete, trie):

```bash
cd src/trie
conda install setup.py
```

## TRF part

TRF training（take aishell for example）：

1. change to egs/aishell
2. add a new example in exp/lm，write config.json and hyper-p.json
   In config.json, scheduler、schedulerz、schedulerd、schedulern means the learning rate of TRF transformer、TRF zeta、 TRF discrete、noise model
   （still has hard code in this part，when changing TRF feature, you need to rewrite scheduler in cat/shared/manager.py（this will be easy after reading the code））
3. run the training
   ```bash
   python utils/pipeline/lm.py exp/task-name
   ```
4. option: 
   --sta :start from which stage，this meaning of stage can be found in utils/pipeline/lm.py, after getting tokenizer, you can start from stage 3
   --ngpu :number of GPU used
5. model saved in exp/task-name/checks
6. you can take exp/lm/trf_best as an example

- TRF testing：
   ```bash
   python utils/pipeline/lm.py exp/task-name --sta 4
   ```
   Start from stage 4，directly test. Will compute the PPL as in hyper-p.json.

- rescoring:
   Can be find in egs/aishell/aishell-rescoring/readme.md. Run cat/lm/rescore.py.
   you can add an option --save-lm-nbest to save the nbest list, choose alpha=1, beta=0.

   
   Weight search：
   example：
   ```bash
      python -m utils.lm.lmweight_search  \
       --weight 1.0 \
       --search 0 1 1\
       --nbestlist aishell-rescoring/rnnt-32_nolm_best-5_aishell-test.nbest aishell-rescoring/nbestlist-LM.nbest aishell-rescoring/nbestlist-L.nbest\
       --ground-truth aishell-rescoring/test.groundtruth\
       --range 0 1 -3 3\
       --cer\
   ```
   Will search the weight of aishell-rescoring/nbestlist-LM.nbest

- generate：
   See in cat/shared/decoder.py  gettrfdata().
