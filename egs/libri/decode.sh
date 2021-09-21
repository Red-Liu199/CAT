#!/bin/bash

dir=$1
weight=$2
beam_size=5
if [ ! $dir ] || [ ! $weight ]; then
    echo "You need to specify the experiment directory and lm weight."
    exit 1
fi

. ./path.sh

echo "Decoding..."
mode='beam'
lmdir="exp/lm1024"
spmdata="data/spm1024"
echo "Settings: mode=$mode | beam-width=$beam_size | lm-weight=$weight"
if [ ! $lmdir ]; then
    echo "You need to specify the extra lm directory."
    exit 1
elif [ ! -d $lmdir ]; then
    echo "No such directory $lmdir"
    exit 1
fi
echo "Ensure modeling unit of transducer is the same as that of extra LM."

dec_dir=$dir/${mode}-${beam_size}-$weight
mkdir $dec_dir || exit 1
mkdir -p $dir/enc
for set in test_clean test_other dev_clean dev_other; do
    echo "> Decoding: $set"
    python3 ctc-crf/parallel_decode.py              \
        --resume=$dir/checks/bestckpt.pt            \
        --config=$dir/config.json                   \
        --input_scp=data/all_ark/$set.scp           \
        --output_dir=$dec_dir                       \
        --enc-out-dir=$dir/enc                      \
        --spmodel=$spmdata/spm.model                \
        --nj=1                                      \
        --mode=$mode                                \
        --beam_size=$beam_size                      \
        --ext-lm-config=$lmdir/lm_config.json       \
        --ext-lm-check=$lmdir/checks/bestckpt.pt    \
        --lm-weight=$weight                         \
        || exit 1

    if [ -f $dec_dir/decode.0-0.tmp ]; then
        cat $dec_dir/decode.?-?.tmp | sort -k 1 > $dec_dir/decode_${set}.txt
        rm $dec_dir/*.tmp
    fi
done

if [ ! $KALDI_ROOT ]; then
    echo "No KALDI_ROOT specified."
    exit 1
fi

echo "" > $dec_dir/result
for set in test_clean test_other dev_clean dev_other; do
    if [ -f $dec_dir/decode_${set}.txt ]; then
        $KALDI_ROOT/src/bin/compute-wer --text --mode=present ark:data/$set/text ark:$dec_dir/decode_${set}.txt | grep WER >> $dec_dir/result
    else
        echo "No decoded text found."
    fi
done

cat $dec_dir/result
