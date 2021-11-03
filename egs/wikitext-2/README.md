# WikiText-2 training example

## Data

Size of datasets (# sequences):

- training set: 25287
- validation set: 3760
- test set: 4358

## Usage

```bash
# view helping info
bash exp/template/run_lm.sh -h

# create a experiment directory and training
mkdir exp/myexp
cp exp/template/config.json exp/myexp/
cp exp/template/run_lm.sh exp/myexp/

bash exp/myexp/run_lm.sh --sta 0
```
