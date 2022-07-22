#!/bin/bash
# author: Huahuan Zheng
set -e -u

<<"PARSER"
("src_trans", type=str, nargs='?',
    default="/mnt/nas3_workspace/spmiData/tasi2_16k/tasi2_8k.txt",
    help="Source data folder containing the audios and transcripts. ")
("-keep-seg", action="store_true", 
    help="Keep original segment spaces, otherwise extra spaces would be removed.")
("-out", type=str, default="./trans.noseg",
    help="Output location to the normalized text.")
PARSER
eval $(python utils/parseopt.py $0 $*)

[ ! -f $src_trans ] && {
    echo "no such trancript file: '$src_trans'"
    exit 1
}

opt_kp=""
[ $keep_seg == "True" ] && opt_kp="--keep-space"

split -d -n l/16 \
    $src_trans ${out}_part_

for x in $(ls ${out}_part_*); do
    sed <$x -e 's/<spn>//g' |
        python local/process_mixing_zh_en.py $opt_kp \
            >$x.done &
done
wait

cat ${out}_part_*.done | sort -k 1,1 -u >$out
rm ${out}_part_*

echo "Text is normalized."
echo "To do further tokenizer training, there're some hints:"
echo "1. Chinese char + English BPE"
echo "   select top-5k most frequent chinese characters and pass to "
echo "   ... SentencePieceTokenizer:user_defined_symbols"
echo ""
echo "2. Phone-based Chinese word + English word"
echo "   You should prepare a lexicon that could cover the whole word list."
echo "   Use JiebaComposeLexiconTokenizer and add your words in lexicon and userdict."
