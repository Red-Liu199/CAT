# WENETSPEECH

## Spectral feature prepare

### Kaldi-freed

1. Download data from [wenet-e2e/WenetSpeech: A 10000+ hours dataset for Chinese speech recognition (github.com)](https://github.com/wenet-e2e/WenetSpeech);
2. Spectral feature generation

    ```bash
    cd ../../tools/CAT/egs/wenetspeech
    sh ./data.sh
    ```

    Defaultly, I use `train-m` subset (~1000 hours) of wenetspeech, which can be specified in the `data.sh`.

### CTC-CRF & Kaldi-styled

1. This is the same as above.
2. Spectral feature gen, dictionary and denominator LM preparation.
    ```bash
    export cwd=$(PWD)
    cd ../../tools/CAT/egs/wenetspeech
    ./run.sh --stage 7
    # pickle data will be stored at data/pickle/*.pickle
    cd $cwd
    cp ../../tools/CAT/egs/wenetspeech/data/pickle/tr.pickle exp/<myexp>/pkl/train.pkl
    cp ../../tools/CAT/egs/wenetspeech/data/pickle/cv.pickle exp/<myexp>/pkl/dev.pkl
    ```
3. Pickle data transform

    Since the pickle data processed with CAT pipeline includes the path weight, which is useless for current repository, we need to remove path weight in the pickle data. This is an example python script:

    ```python
    import pickle
    expdir='exp/<myexp>'
    for _set in ['train', 'dev']:
        f_pkl = f"{expdir}/pkl/{_set}.pkl"
        with open(f_pkl, 'rb') as fi:
            data = pickle.load(fi)

        for i in range(len(data)):
            # path weight is the last item
            data[i].pop()

        with open(f_pkl, 'wb') as fo:
            pickle.dump(data, fo)
    ```


## Training and evaluation

### Kaldi-freed training

```bash
python utils/asr_process.py exp/<myexp>
```

## Kaldi-styled training

```bash
# conduct nn training only
python utils/asr_process.py exp/<myexp> --sta 3 --sto 3
```
After NN training finished, you need to manually proceed the decoding.

1. generate logits, for more details

    ```bash
    python -m cat.ctc.cal_logit -h
    ```

2. decode with FST.

    ```bash
    cd ../../tools/CAT/egs/wenetspeech/
    # set logit path in run.sh stage 8, then run
    ./run.sh --stage 8
    ```