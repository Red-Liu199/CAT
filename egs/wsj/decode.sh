#!/bin/bash

dir=$1
spmdata="data/spm"
if [ ! $dir ]; then
    echo "You need to specify the experiment directory."
    exit 1
fi

echo "Decoding..."
mode='beam'
beam_size=5
echo "Settings: mode=$mode | beam-width=$beam_size"

dec_dir=$dir/decode-${mode}-${beam_size}
mkdir -p $dec_dir
for set in eval92 dev93; do
    python3 ctc-crf/transducer_decode.py    \
        --resume=$dir/ckpt/bestckpt.pt      \
        --config=$dir/config.json           \
        --input_scp=data/all_ark/$set.scp   \
        --output_dir=$dec_dir               \
        --spmodel=$spmdata/spm.model        \
        --mode=$mode                        \
        --beam_size=$beam_size              \
        || exit 1

    if [ -f $dec_dir/decode.0.tmp ]; then
        cat $dec_dir/decode.?.tmp | sort -k 1 > $dec_dir/decode_${set}.txt
        rm $dec_dir/*.tmp
    fi
done

echo "" > $dec_dir/result
for set in eval92 dev93; do
    if [ -f $dec_dir/decode_${set}.txt ]; then
        $KALDI_ROOT/src/bin/compute-wer --text --mode=present ark:data/test_$set/text ark:$dec_dir/decode_${set}.txt >> $dec_dir/result
        echo "" >> $dec_dir/result
    else
        echo "No decoded text found."
    fi
done

cat $dec_dir/result
