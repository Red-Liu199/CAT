## TRF LM

read instructions from Haochen Liu [intro](../../../aishell/exp/archive/trf/readme_trf.md).

To train a TRF LM, please follow these steps (currently unusable):

0. Download text data.

    ```bash
    bash local/lm_data.sh
    ```

1. Train a NN LM as the noise model. (This would takes several minutes depending on your GPU model.)

    ```bash
    python utils/pipeline/lm.py exp/lm-nn --ngpu 1
    ```

2. Train the tokenizer and process the data

    ```bash
    # --sto 2 means ending at stage 2
    python utils/pipeline/lm.py exp/lm-trf --sto 2
    ```

3. Prepare the length information and discrete feats (This takes several minutes, and you will see a progress bar.)

    ```bash
    mkdir -p exp/lm-trf/trf_cache
    python -m cat.lm.trf.prep_feats exp/lm-trf/lmbin/train.pkl exp/lm-trf/trf_cache/linfo.pkl --feat_type_file exp/lm-trf/grammar.fs --f_feats exp/lm-trf/trf_cache/4gram.fs
    ```

    When jobs finished, there'll be two new files in `exp/lm-trf/trf_cache/`: `linfo.pkl` and `4gram.fs`

4. Train the discrete features with TRF. (This step is bugging now. The code review hurts my head.)

    ```bash
    # --sta 3 means starting from stage 3
    CUDA_VISIBLE_DEVICES=0 python utils/pipeline/lm.py exp/lm-trf --sta 3 --sto 3 --ngpu 1
    ```
