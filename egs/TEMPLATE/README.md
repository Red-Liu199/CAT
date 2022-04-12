# TEMPLATE

## Transducer (RNN-T)

1. Prepare data.

   ```bash
   bash local/data.sh
   ```

2. Train Transducer

   ```bash
   python utils/pipeline_asr.py exp/template-rnnt --ngpu 1
   ```

## CTC

1. Prepare data.

   ```bash
   bash local/data.sh
   ```

2. Train CTC

   ```bash
   python utils/pipeline_asr.py exp/template-ctc --ngpu 1
   ```

## Neural language model (NN LM)

1. Prepare data.

   ```bash
   bash local/lm_data.sh
   ```

2. Train a Transformer LM

   ```bash
   python utils/pipeline_lm.py exp/template-lm-nn --ngpu 1
   ```


## N-gram LM

1. Prepare data.

   ```bash
   bash local/lm_data.sh
   ```

2. Train a 3-gram word LM

   ```bash
   bash utils/pipeline_ngram.sh exp/template-lm-ngram --text-corpus -o 3
   ```