#!/bin/bash
# author: Huahuan Zheng
# This script shows how to train a CTC-CRF model.
set -e

KALDI_ROOT=/opt/kaldi
DIR=$(dirname $0)

[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set."
    exit 1
}

# prepare data
bash local/data.sh

# prepare tokenizer
python utils/pipeline/asr.py $DIR --sto 1

# get trancript corpus from hyper-p.json:data:train
f_text=$(cat $DIR/hyper-p.json | python -c "
import sys,json
from cat.utils.pipeline.asr import resolve_in_priority 
files = ' '.join(sum(resolve_in_priority(json.load(sys.stdin)['data']['train']), []))
print(files)")
for x in $f_text; do
    [ ! -f $x ] && echo "No such training corpus: '$x'" && exit 1
done

# prepare den lm
cat $f_text | bash utils/tool/prep_den_lm.sh \
    -tokenizer="$DIR/tokenizer.tknz" \
    -kaldi-root=$KALDI_ROOT \
    -ngram-order=3 \
    -no-prune-ngram-order=2 \
    /dev/stdin $DIR/den_lm.fst

echo "Denominator LM stored at $DIR/den_lm.fst"

# prepare decode lm
bash utils/pipeline/ngram.sh $DIR/decode-lm -o 3

# finish rest stages
python utils/pipeline/asr.py $DIR --sta 2 --ngpu 1

echo "$0 done."
exit 0
