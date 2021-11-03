#!/bin/bash
set -u
opts=$(python utils/parseopt.py '{
        "nbestlist":{
            "type": "str",
            "help": "N-best list file."
        },
        "spmodel":{
            "type": "str",
            "help": "Sentence piece model location."
        },
        "gt_text":{
            "type": "str",
            "help": "Location of ground truth label."
        },
        "lmdir":{
            "type": "str",
            "help": "Language model directory. Used at lamb > 0. Supposed to include <lmdir>/config.json and <lmdir>/checks/<check>"
        },
        "--check":{
            "type": "str",
            "default": "bestckpt.pt",
            "help": "Name of checkpoint. Default: bestckpt.pt"
        },
        "--lamb":{
            "type":"float",
            "default": 0.1,
            "help": "External language model weight. Default: 0.1 (disable)"
        },
        "--cpu":{
            "action":"store_true",
            "help": "Use cpu to decode. Default: False"
        },
        "--nj":{
            "type":"int",
            "default": -1,
            "help": "Number of threads when using CPU. Default: -1 (all available)."
        },
        "--cer":{
            "action":"store_true",
            "help": "Compute CER along with WER. Default: False"
        }
    }' $0 $*) && eval $opts || exit 1

if [ -d $spmodel ]; then
    export spmodel=$spmodel/spm.model
fi
if [ $nj -eq "-1" ]; then
    nj=$(nproc)
fi

if [ $cpu == "True" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

checkpoint=$lmdir/checks/$check
python utils/checkfile.py -f $gt_text $checkpoint $lmdir/config.json $nbestlist $spmodel || exit 1

python -m cat.lm.rescore \
    $nbestlist ./rescore.out.tmp \
    --dist-url="tcp://127.0.0.1:13245" \
    --lm-config=$lmdir/config.json \
    --lm-check=$checkpoint \
    --spmodel=$spmodel \
    --nj=$nj \
    --lamb=$lamb ||
    exit 1

if [ -f ./rescore.out.tmp ]; then
    cat ./rescore.out.tmp | sort -k 1 >./rescore.sorted.out.tmp || exit 1
    rm ./rescore.out.tmp
fi

if [ -f ./rescore.sorted.out.tmp ]; then
    if [ $cer == "True" ]; then
        python utils/wer.py --stripid --cer $gt_text ./rescore.sorted.out.tmp || exit 1
    else
        python utils/wer.py --stripid $gt_text ./rescore.sorted.out.tmp || exit 1
    fi
fi
