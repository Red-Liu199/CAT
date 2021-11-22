# Copyright Tsinghua University 2021
# Author: Huahuan Zheng (maxwellzh@outlook.com)
set -u
opts=$(python utils/parseopt.py '{
        "dir":{
            "type": "str",
            "help": "Path to the working directory."
        },
        "outlm":{
            "type":"str",
            "default": "ngram.klm",
            "help": "Name of output N-gram file. Default: ngram.klm"
        },
        "-o":{
            "type":"int",
            "default": 5,
            "dest": "order",
            "help": "Max order of n-gram. default: 5"
        },
        "--arpa":{
            "action": "store_true",
            "help": "Store n-gram file as .arpa instead of binary."
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

textbin=$dir/lmbin/train.pkl
if [ ! -f $textbin ]; then
    python utils/lm_process.py $dir --sta 2 --sto 2 || exit 1
else
    echo "$textbin found, skip generating."
fi

if [ $arpa == "True" ]; then
    python utils/readtextbin.py $textbin |
        lmplz -o $order -S 80% --discount_fallback >$outlm
else
    python utils/readtextbin.py $textbin |
        lmplz -o $order -S 80% --discount_fallback |
        build_binary /dev/stdin $outlm
fi
