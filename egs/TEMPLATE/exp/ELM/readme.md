# Train Energy-based Language Model with Dynamic NCE
To train an ELM, please follow these steps:

0. Download text data.
    ```bash
    bash local/lm_data.sh
    ```

1. Train the tokenizer and process the data

    ```bash
    # --sto 2 means ending at stage 2
    python utils/pipeline/lm.py exp/ELM --sto 2
    ```

2. Do some additional processing to the data

    ```bash
    mv exp/ELM/pkl exp/ELM/pkl0
    python utils/reprocess.py exp/ELM/pkl0 exp/ELM/pkl1 --head_del 1
    ```

4. Train and test the ELM with DNCE.

    ```bash
    python utils/pipeline/lm.py exp/ELM --start 3 --ngpu 4
    ```