#!/bin/bash
# Author: Sunny
set -e -u

dir="exp/asr-ctc-crf-phone"
KALDI_ROOT=/opt/kaldi
export KALDI_ROOT=$KALDI_ROOT

# prepare den_lm.fst
[ ! -f $dir/den_lm.fst ] &&
DIR=$(dirname $0)
d_text=data/src/train/text
         bash utils/tool/prep_den_lm.sh \
            -tokenizer="$DIR/tokenizer.tknz" \
            -kaldi-root=$KALDI_ROOT \
            $d_text $DIR/den_lm.fst
echo "den_lm.fst finsh" 

# model training
python utils/pipeline/asr.py $dir

# prepare decoding graph
lm="$dir/decode_lm/4gram.arpa"
bash utils/pipeline/ngram.sh $dir/decode_lm \
    -o 4 --arpa --output $lm --stop_stage 3
echo "graph finsh"

# prepare decoding TLG.fst
function get_tokenizer() {
    echo $(
        python -c \
            "import json;print(json.load(open('$1/hyper-p.json'))['tokenizer']['file'])"
    )
}

bash utils/tool/build_decoding_graph.sh \
    $(get_tokenizer $dir) \
    $(get_tokenizer $dir/decode_lm) \
    $lm $dir/graph

echo "TLG.fst finsh"

# TLG decoding 
for set in dev93 eval92; do
bash  cat/ctc/fst_decode.sh \
    --lmwt 1.0 --acwt 1.0 --lattice-beam 8.0 \
    $dir/decode/${set} $dir/graph $dir/decode/${set}
done

# eval wer

python utils/wer.py $(find data -iname "eval*.text") $dir/decode/eval92/text_*
python utils/wer.py $(find data -iname "dev93*.text") $dir/decode/dev93/text_*

exit 0
