#!/bin/bash
# Copyright Tsinghua University 2021
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# Script for training n-gram LM
set -u
set -e
<<"PARSER"
("dir", type=str, help="Path to the LM directory.")
("--start-stage", type=int, default=1, help="Start stage of the script.")
("-o", "--order", type=int, default=5, help="Max order of n-gram. default: 5")
("--output", type=str, default="$dir/${order}gram.klm",
    help="Path of output N-gram file. default: [dir]/[order]gram.klm")
("--arpa", action="store_true", help="Store n-gram file as .arpa instead of binary.")
("--large-corpus", action="store_true", help="Use on-the-fly encoding for large corpus.")
("--prune", type=str, default="", nargs='*',
    help="Prune options passed to KenLM lmplz executable. default: ")
PARSER
opts=$(python utils/parseopt.py $0 $*) && eval $opts || exit 1

export PATH=$PATH:../../src/bin/
[ ! $(command -v lmplz) ] && echo "command not found: lmplz" && exit 1
[ ! $(command -v build_binary) ] && echo "command not found: build_binary" && exit 1
[ ! -d $dir ] && echo "No such directory: $dir" && exit 1

# train sentence piece tokenizer
[ $start_stage -le 1 ] && python utils/lm_process.py $dir --sta 1 --sto 1

# we need to manually rm the bos/eos/unk since lmplz tool would add them
# and kenlm not support <unk> in corpus,
# ...so in the `utils/readtextbin.py` script we convert 0(<bos>, <eos>) and 1 (<unk>) to white space
# ...if your tokenizer set different bos/eos/unk id, you should make that mapping too.
export tokenizer="$(cat $dir/hyper-p.json |
    python -c "import sys,json;print(json.load(sys.stdin)['tokenizer']['location'])")"
if [ $large_corpus == "True" ]; then
    f_text=$(cat $dir/hyper-p.json |
        python -c "import sys,json;print(json.load(sys.stdin)['data']['train'])" |
        sed "s/\[//g" | sed "s/\]//g" | sed "s/'//g" | sed "s/,/ /g")

    [ ! -f $tokenizer ] && echo "No tokenizer model: '$tokenizer'" && exit 1
    for x in $f_text; do
        [ ! -f $x ] && echo "No such training corpus: '$x'" && exit 1
    done

    processing="cat $f_text | python utils/readtextbin.py . -t --tokenizer $tokenizer --map 0: 1:"
else
    textbin=$dir/lmbin/train.pkl
    if [ ! -f $textbin ]; then
        python utils/lm_process.py $dir --sta 2 --sto 2 || exit 1
    else
        echo "$textbin found, skip generating."
    fi
    [ ! -f $textbin ] && echo "No binary text file: '$textbin'" && exit 1
    processing="python utils/readtextbin.py $textbin --map 0: 1:"
    unset textbin
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
if [ -f $dir/config.json ]; then
    cat $dir/config.json | python -c "
import sys, json
configure = json.load(sys.stdin)
configure['decoder']['kwargs']['f_binlm'] = '$output'
configure['decoder']['kwargs']['gram_order'] = $order
from cat.shared import tokenizer as tknz
tokenizer = tknz.load('$tokenizer')
configure['decoder']['kwargs']['num_classes'] = tokenizer.vocab_size
json.dump(configure, sys.stdout, indent=4)" >$dir/config.json.tmp
    mv $dir/config.json.tmp $dir/config.json
fi

# test
# You may need to set the 'num_classes' in
# ... $dir/config.json to the number of vocab of your TOKENIZER
python utils/ppl_compute_ngram.py $dir
