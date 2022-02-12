# WikiText-2 training example

## Data

Size of datasets (# sequences):

- training set: 25287
- validation set: 3760
- test set: 4358

## Usage

```bash
# generate normalized data
python local/data_prep.py

# train NN LM
python utils/lm_process.py exp/template-word-transformer/
# train N-gram LM
./utils/ngram_process.sh exp/template-word-ngram/
```
