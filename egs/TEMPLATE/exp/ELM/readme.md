# Train Energy-based Language Model with Dynamic NCE
To train an ELM, please follow these steps:

0. Download text data.
    ```bash
    bash local/lm_data.sh
    ```

1. Train the tokenizer and process the data

    ```bash
    python utils/pipeline/lm.py exp/ELM 
    ```