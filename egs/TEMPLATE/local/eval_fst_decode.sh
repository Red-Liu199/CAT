#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
set -e -u
<<"PARSER"
("dir", type=str, help="Experiment directory.")
("graph", type=str, 
    help="Decoding graph directory, where TLG.fst and r_words.txt are expected.")
("--data", type=str, nargs='+', required=True,
    help="Dataset(s) to be evaluated. e.g. dev test")
("--acwt", type=float, default=1.0, 
    help="AC score weight.")
("--lmwt", type=float, default=0.2, 
    help="LM score weight.")
("--wip", type=float, default=0.0, 
    help="Word insertion penalty.")
PARSER
eval $(python utils/parseopt.py $0 $*)

cache="/tmp/$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 13
    echo ''
).log"
for set in $data; do
    fout=$(bash cat/ctc/fst_decode.sh \
        --acwt $acwt \
        --lmwt $lmwt \
        --wip $wip \
        $dir/decode/$set $graph $dir/decode/$set)

    echo -en "$fout\t"
    python utils/wer.py --cer \
        data/src/$set/text $fout

done 2>$cache || {
    cat $cache
    exit 1
}
rm $cache

exit 0
