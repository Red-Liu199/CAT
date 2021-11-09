# WikiText-2 training example

## Data

Size of datasets (# sequences):

- training set: 25287
- validation set: 3760
- test set: 4358

## Usage

```bash
# view helping info
python utils/lm_process.py -h

# create a experiment directory from template and training
mkdir exp/myexp
cp exp/template/*.json exp/myexp/

python utils/lm_process.py exp/myexp/
```
