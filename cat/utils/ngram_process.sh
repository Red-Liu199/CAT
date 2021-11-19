# Copyright Tsinghua University 2021
# Author: Huahuan Zheng (maxwellzh@outlook.com)
set -u
opts=$(python utils/parseopt.py '{
        "textbin":{
            "type": "str",
            "help": "Path to the tokenized text file (normally *.pkl)"
        },
        "export":{
            "type":"str",
            "default": "./",
            "help": "Path to outout file. Usually in format .klm"
        },
        "-o":{
            "type":"int",
            "default": 5,
            "dest": "order",
            "help": "Max order of n-gram. default: 5"
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

python utils/readtextbin.py $textbin |
    lmplz -o $order -S 80% --discount_fallback |
    build_binary /dev/stdin $export
