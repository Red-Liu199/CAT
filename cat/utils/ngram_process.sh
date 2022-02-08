# Copyright Tsinghua University 2021
# Author: Huahuan Zheng (maxwellzh@outlook.com)
set -u
set -o errexit
opts=$(python utils/parseopt.py '{
        "dir":{
            "type": "str",
            "help": "Path to the working directory."
        },
        "--output":{
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
        "--prune":{
            "type": "str",
            "default": " ",
            "nargs": "*",
            "help": "Prune options passed to KenLM lmplz executable. Default: "
        }
    }' $0 $*) && eval $opts || exit 1
# argument parsed dnoe

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

if [ ! -d $(dirname $output) ]; then
    output=$dir/$output
fi

# train sentence piece tokenizer
python utils/lm_process.py $dir --sta 1 --sto 1 || exit 1

# we need to manually rm the bos/eos/unk since lmplz tool would add them
# and kenlm not support <unk> in corpus,
# ...so in the `utils/readtextbin.py` script we convert 0(<unk>) and 1 (<unk>) to white space
# ...if your tokenizer set different bos/eos/unk id, you should make that mapping too.
if [ $large_corpus == "True" ]; then
    spmodel=$(cat $dir/hyper-p.json | python -c "import sys;import json;print(json.load(sys.stdin)['sp']['model_prefix'])").model
    f_text=$(cat $dir/hyper-p.json | python -c "import sys;import json;print(json.load(sys.stdin)['data']['train'])")

    if [ ! -f $f_text ] || [ ! -f $spmodel ]; then
        echo "Make sure '$f_text' and '$spmodel' exist."
        exit 1
    fi
    processing="cat $f_text | python utils/readtextbin.py . -t --spm $spmodel --map 0: 1:"
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
    processing="python utils/readtextbin.py $textbin --map 0: 1:"
fi

if [ "$prune" ]; then
    prune="--prune $prune"
fi

if [ $arpa == "True" ]; then
    eval "$processing | 
        lmplz -o $order $prune -S 80% --discount_fallback >$output"
else
    eval "$processing |
        lmplz -o $order $prune -S 80% --discount_fallback |
        build_binary /dev/stdin $output"
fi
echo "LM saved at $output"
echo ""

# test
echo "N-gram LM training is finished. Use following command to evaluate model performance."
echo "python utils/ppl_compute_ngram.py $dir"
