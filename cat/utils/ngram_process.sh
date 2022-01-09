# Copyright Tsinghua University 2021
# Author: Huahuan Zheng (maxwellzh@outlook.com)
set -u
set -o errexit
opts=$(python utils/parseopt.py '{
        "dir":{
            "type": "str",
            "help": "Path to the working directory."
        },
        "--outlm":{
            "type": "str",
            "default": "ngram.klm",
            "help": "Name of output N-gram file. Default: ngram.klm"
        },
        "-o":{
            "type": "int",
            "default": 5,
            "dest": "order",
            "help": "Max order of n-gram. default: 5"
        },
        "--arpa":{
            "action": "store_true",
            "help": "Store n-gram file as .arpa instead of binary."
        },
        "--large-corpus":{
            "action": "store_true",
            "help": "Use on-the-fly encoding for large corpus."
        },
        "--opts-ngram":{
            "type": "str",
            "default": " ",
            "help": "Custom options passed to KenLM lmplz executable. Default: "
        }
    }' $0 $*) && eval $opts || exit 1

export PATH=$PATH:../../src/bin/
if [ ! $(command -v lmplz) ]; then
    echo "lmplz is not found"
    exit 1
fi
if [ ! $(command -v build_binary) ]; then
    echo "build_binary is not found"
    exit 1
fi

if [ ! -d $dir ]; then
    echo "No such directory: $dir"
    exit 1
fi

if [ ! -d $outlm ]; then
    outlm=$dir/$outlm
fi

# train sentence piece tokenizer
python utils/lm_process.py $dir --sta 1 --sto 1 || exit 1

if [ $large_corpus == "True" ]; then
    spmodel=$(cat $dir/hyper-p.json | python -c "import sys;import json;print(json.load(sys.stdin)['sp']['model_prefix'])").model
    f_text=$(cat $dir/hyper-p.json | python -c "import sys;import json;print(json.load(sys.stdin)['data']['train'])")

    if [ ! -f $f_text ] || [ ! -f $spmodel ]; then
        echo "Make sure '$f_text' and '$spmodel' exist."
        exit 1
    fi
    processing="cat $f_text | python utils/readtextbin.py . -t --spm $spmodel"
else
    textbin=$dir/lmbin/train.pkl
    if [ ! -f $textbin ]; then
        python utils/lm_process.py $dir --sta 2 --sto 2 || exit 1
    else
        echo "$textbin found, skip generating."
    fi
    if [ ! -f $textbin ]; then
        echo "Make sure '$textbin' exists."
        exit 1
    fi
    processing="python utils/readtextbin.py $textbin"
fi

if [ $arpa == "True" ]; then
    eval "$processing |
        lmplz -o $order $opts_ngram -S 80% --discount_fallback >$outlm"
else
    eval "$processing |
        lmplz -o $order $opts_ngram -S 80% --discount_fallback |
        build_binary /dev/stdin $outlm"
fi

# test
python utils/ppl_compute_ngram.py $dir