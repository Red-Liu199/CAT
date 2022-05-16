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
("--text-corpus", action="store_true", help="Use on-the-fly encoding for text corpus.")
("--prune", type=str, default="", nargs='*',
    help="Prune options passed to KenLM lmplz executable. default: ")
("--type", type=str, default="trie", choices=['trie', 'probing'],
    help="Binary file structure. default: trie")
PARSER
opts=$(python utils/parseopt.py $0 $*) && eval $opts || exit 1

export PATH=$PATH:../../src/bin/
[ ! $(command -v lmplz) ] && echo "command not found: lmplz" && exit 1
[ ! $(command -v build_binary) ] && echo "command not found: build_binary" && exit 1
[ ! -d $dir ] && echo "No such directory: $dir" && exit 1

# train sentence piece tokenizer
[ $start_stage -le 1 ] && python utils/pipeline/lm.py $dir --sta 1 --sto 1
[ ! -f $dir/hyper-p.json ] && echo "No hyper-setting file: $dir/hyper-p.json" && exit 1
[ ! -f $dir/config.json ] && echo "No model config file: $dir/config.json" && exit 1

if [ "$prune" ]; then
    prune="--prune $prune"
fi

export text_out="/tmp/$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 13
    echo ''
).corpus.tmp"
export arpa_out="/tmp/$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 13
    echo ''
).arpa.tmp"

# we need to manually rm the bos/eos/unk since lmplz tool would add them
# and kenlm not support <unk> in corpus,
# ...so in the `utils/data/readtextbin.py` script we convert 0(<bos>, <eos>) and 1 (<unk>) to white space
# ...if your tokenizer set different bos/eos/unk id, you should make that mapping too.
export tokenizer="$(cat $dir/hyper-p.json |
    python -c "import sys,json;print(json.load(sys.stdin)['tokenizer']['location'])")"
if [ $text_corpus == "True" ]; then
    f_text=$(cat $dir/hyper-p.json |
        python -c "import sys,json;print(json.load(sys.stdin)['data']['train'])" |
        sed "s/\[//g" | sed "s/\]//g" | sed "s/'//g" | sed "s/,/ /g")

    [ ! -f $tokenizer ] && echo "No tokenizer model: '$tokenizer'" && exit 1
    for x in $f_text; do
        [ ! -f $x ] && echo "No such training corpus: '$x'" && exit 1
    done

    python utils/data/readtextbin.py $f_text -o $text_out \
        -t --tokenizer $tokenizer --map 0: 1:
else
    textbin=$dir/lmbin/train.pkl
    if [ ! -f $textbin ]; then
        python utils/pipeline/lm.py $dir --sta 2 --sto 2 || exit 1
    else
        echo "$textbin found, skip generating."
    fi

    [ ! -f $textbin ] && echo "No binary text file: '$textbin'" && exit 1
    python utils/data/readtextbin.py $textbin \
        -o $text_out --map 0: 1:
fi

train_cmd="lmplz <$text_out -o $order $prune -S 20%"
[ $arpa == "True" ] && train_cmd="$train_cmd >$output"

# NOTE: if lmplz raises error telling the counts of n-grams are not enough,
# you should probably duplicate your text corpus or add the option --discount_fallback
# Error msg sample:
# "ERROR: 3-gram discount out of range for adjusted count 3: -5.2525253."
train_cmd="$train_cmd || $train_cmd --discount_fallback"

[ $arpa == "False" ] && train_cmd="($train_cmd) | build_binary $type /dev/stdin $output"
eval $train_cmd

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

echo "LM saved at $output."

[ ! -f $dir/readme.md ] && (
    echo -e "\ntrain command:\n" >>$dir/readme.md
    echo -e "\`\`\`bash\n$0 $@\n\`\`\`" >>$dir/readme.md
    echo -e "\nproperty:\n" >>$dir/readme.md
    echo "- prune: $prune" >>$dir/readme.md
    echo "- type:  $type" >>$dir/readme.md
    echo "- size:  $(ls -lh $output | awk '{print $5}')B" >>$dir/readme.md
    echo -e "\nperplexity:\n" >>$dir/readme.md
    echo -e "\`\`\`\n\n\`\`\`" >>$dir/readme.md
)

python utils/pipeline/lm.py $dir --start_stage 4
exit 0
