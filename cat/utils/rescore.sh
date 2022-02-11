#!/env/sh
# Copyright 2021 Tsinghua University,
# Author Huahuan Zheng (maxwellzh@outlook.com)
#
# script to rescore the n-best list
set -u
set -e
<<"PARSER"
{
    "lmdir": {
        "type": "str",
        "help": "Directory of language model."
    },
    "set": {
        "type": "str",
        "help": "Name of evaluation set, shoud be match with N-best list."
    },
    "nbestlist": {
        "type": "str",
        "help": "Path to the file of N-best list."
    },
    "--alpha": {
        "type": "float",
        "default": 1.0,
        "help": "LM weight of rescoring. Default: 1.0"
    },
    "--beta": {
        "type": "float",
        "default": 0.0,
        "help": "Token insert penalty. Default: 0.0"
    },
    "--sp": {
        "type": "str",
        "help": "Path to the SentencePiece model. If not set, would try to resolve from ./sentencepiece/"
    }
}
PARSER
opts=$(python utils/parseopt.py $0 $*) && eval $opts || exit 1

[ ! -d $lmdir ] && echo "No such LM dir $lmdir" && exit 1
[ ! -f $nbestlist ] && echo "No such N-best list file $nbestlist" && exit 1
[ ! -f $lmdir/config.json ] && echo "Missing LM configuration $lmdir/config.json" && exit 1
[ ! -f $lmdir/checks/best-10.pt ] && echo "Missing LM checkpoint $lmdir/checks/best-10.pt. This is OK if you're using N-gram model."
if [ $sp == "None" ]; then
    # resolve
    echo "--sp is not specified, try to resolve from ./sentencepiece/"
    if [[ -d sentencepiece ]] && [[ "$(ls sentencepiece/**/*.model)" ]]; then
        export models="$(ls sentencepiece/**/*.model)"
        if [ $(echo "$models" | wc -w) -gt 1 ]; then
            echo "Found more than one SP model: $models"
            echo "you should choose one and pass it via --sp" && exit 1
        else
            echo "Found SP model: $models"
            export sp=$models
        fi
        unset models
    else
        echo "No model is found." && exit 1
    fi
fi
[ ! -f $sp ] && echo "No such SentencePiece model $sp" && exit 1

cachefile="/tmp/$(md5sum $nbestlist | cut -d ' ' -f 1)"
python -m cat.lm.rescore \
    --config=$lmdir/config.json \
    --resume=$lmdir/checks/best-10.pt \
    --nj=20 \
    --alpha=$alpha \
    --beta=$beta \
    --spmodel=$sp \
    --verbose \
    $nbestlist $cachefile || exit 1

echo -ne "$set    "
python utils/wer.py \
    --stripid \
    ../../tools/CAT/egs/libri/data/$set/text \
    $cachefile

rm $cachefile
