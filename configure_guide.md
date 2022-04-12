# Configuration

Basically, I use two files to control the whole pipeline of data-preparation / tokenizer training / model training / evaluation, which commonly look like

```
egs/<task>/exp/template
├── config.json
└── hyper-p.json
```

## Hyper-parameter configuration

`hyper-p.json`, example taken from `egs/libri`

```
{
    // data pre-processing related
    "data": {
        "train": ...,
        "dev": ...,
        "test": ...,
        // "filter" is for ASR task only, filter out utterances shorter than 10 and longer than 2000 (frames)
        "filter": "10:2000",
        // "text_processing" is for LM task only. code: utils/transText2Bin.py
        "text_processing": {
            // truncate the utterances by 128 (tokens)
            "truncate": 128
        }
    },
    "tokenizer": {
        "type": ...,        // can be any class derived from 'AbsTokenizer' in cat/shared/tokenizer.py
        "location": ...     // Path to the tokenizer file.
        "property": ...     // keyword arguments for training the tokenizer
                            // for SentencePiece tokenizer, please refer to:
                            // https://github.com/google/sentencepiece/blob/master/doc/options.md
    },
    // NN training related setting, for supported options (in egs/<task>/):
    // RNN-T: run 'python -m cat.rnnt -h'
    // CTC: run 'python -m cat.ctc -h'
    // LM: run 'python -m cat.lm -h'
    "train": {
        ...
    },
    // inference related setting
    "inference": {
        // model averaging setting, optional
        "avgmodel": {
            "mode": "best",  // 'best' of 'last'
            "num": 10        // number of checkpoints to be averaged
        },
        // perplexity computation for LM, for support optionsrun (in egs/<task>/):
        // run 'python -m cat.lm.ppl_compute -h'
        "ppl": {
            ...
        },
        // ASR only, decoding setting, for support options (in egs/<task>/):
        // RNN-T: run 'python -m cat.rnnt.decode -h'
        // CTC: run 'python -m cat.ctc.decode -h'
        "decode": {
            ...
        },
        // ASR only, WER/CER computing setting, run `python utils/wer.py -h` for more options
        "er": {
            "mode": "wer",  // 'wer' or 'cer'
            "oracle": true  // compute oracle wer for N-best list or not
        }
    },
    // the git commit hash, useful to reproduce the experiment
    "commit": "60aa5175c9630bcb5ea1790444732fc948b05865"
}
```

## Model configuration

`config.json`, example taken from `egs/libri`

```
{
    // for ASR only, code: cat/shared/_specaug.py
    "specaug_config": {
        ...
    },
    // required for CRF, optional for CTC, code: `build_model()` in cat/ctc/train.py
    "ctc-trainer": {
        "use_crf": false,           // enable CRF loss or not, if false, following two options would be useless.
        "lamb": 0.01,               // weight of CTC loss once enable CRF loss
        "den-lm": "/path/to/den_lm" // location of denominator LM
    },
    // required for RNN-T, code: class 'TransducerTrainer' in cat/rnnt/train.py
    "transducer": {
        ...
    },
    // required for RNN-T, code: cat/rnnt/joint.py
    "joint": {
        "type": ...,   // can be any class derived from 'AbsJointNet' in cat/rnnt/joint.py
        "kwargs": {    // arguments according to 'type'
            ...
        }
    },
    // required for ASR task, code: cat/shared/encoder.py
    "encoder": {
        "type": ...,   // can be any class derived from 'AbsEncoder' in cat/shared/encoder.py
        "kwargs": {    // arguments according to 'type'
            ...
        }
    },
    // required for both RNN-T and LM, code: cat/shared/decoder.py 
    "decoder": {
        "type": ...,   // can be any class derived from 'AbsDecoder' in cat/shared/decoder.py 
        "kwargs": {    // arguments according to 'type'
            ...
        }
    },
    // scheduler settings, required for all NN model training. code: cat/shared/scheduler.py
    "scheduler": {
        "type": ...,   // can be any class derived from `Scheduler` in cat/shared/scheduler.py
        "kwargs": {    // arguments according to 'type'
            ...
        },
        // optimizer settings
        "optimizer": {
            "type": ...,       // all available ones in torch.optim
            "use_zero": true,  // flag of whether use 'ZeroRedundancyOptimizer' for less memory usage.
            "kwargs": {        // arguments according to 'type'
                ...
            }
        }
    }
}
```

## Usage with Visual Studio Code

If you're using [Visual Studio Code](https://code.visualstudio.com/) as working environment, you can setup the json schema for syntax intellisense via (in `egs/<task>/`):

```bash
ln -s ../../.vscode ./
```

Above command would probably raise an error, if there exists a directory `egs/<task>/.vscode`, in such situation, you could manually copy the schema files

```bash
cp ../../.vscode/{hyper_schema,schemas}.json ./.vscode/
```

And add following contents into the file `egs/<task>/.vscode/settings.json`:

```
{
    ...,        // there might be existing settings
    "json.schemas": [
        {
            "fileMatch": [
                "exp/**/config.json"
            ],
            "url": ".vscode/schemas.json"
        },
        {
            "fileMatch": [
                "exp/**/hyper-p.json"
            ],
            "url": ".vscode/hyper_schema.json"
        }
    ]
}
```

With all these properly setup, intellisense will be enable when editting `egs/<task>/<any name>/config.json` and `egs/<task>/<any name>/hyper-p.json`.

<img src="assets/intellisense.gif" width=800px/>

For more about how schema works, refer to [JSON editing in Visual Studio Code](https://code.visualstudio.com/docs/languages/json).
