#!/bin/bash
# author: Huahuan Zheng
set -e -u

src="/mnt/nas3_workspace/spmiData/tasi2_16k/tasi2_8k.txt"
keep_seg=trued
f_out="trans.noseg"

[ ! -f $src ] && {
    echo "no such trancript file: '$src'"
    exit 1
}

opt_kp=""
[ $keep_seg == "true" ] && opt_kp="--keep-space"

split -d -n l/16 \
    $src ${f_out}_part_

for x in $(ls ${f_out}_part_*); do
    sed <$x -e 's/<spn>//g' |
        python process_mixing_zh_en.py $opt_kp \
            >$x.done &
done
wait

cat ${f_out}_part_*.done | sort -k 1,1 -u >$f_out
rm ${f_out}_part_*

echo "Text is normalized."
echo "To do further tokenizer training, there're some hints:"
echo "1. Chinese char + English BPE"
echo "   select top-5k most frequent chinese characters and pass to "
echo "   ... SentencePieceTokenizer:user_defined_symbols"
echo ""
echo "2. Phone-based Chinese word + English word"
echo "   You should prepare a lexicon that could cover the whole word list."
echo "   Use JiebaComposeLexiconTokenizer and add your words in lexicon and userdict."
