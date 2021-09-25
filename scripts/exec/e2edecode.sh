#!/bin/bash

opts=$(python exec/parseopt.py '{
        "dir":{
            "type": "str",
            "help": "Directory of experiment."
        },
        "test_set":{
            "type": "str",
            "help": "Test sets split by ':'. Such as: \"eval92:dev93\"."
        },
        "spmodel":{
            "type": "str",
            "help": "Sentence piece model location."
        },
        "--lm_weight":{
            "type":"float",
            "default": 0.0,
            "help": "External language model weight. Default: 0.0 (disable)"
        },
        "--lmdir":{
            "type": "str",
            "help": "Language model directory. Used at lm_weight > 0. Supposed to include <lmdir>/lm_config.json and <lmdir>/checks/bestckpt.pt"
        },
        "--beam_size":{
            "type":"int",
            "default": 5,
            "help": "Beam search width. Default: 5"
        }
    }' $0 $*) && eval $opts || exit 1

. ./path.sh

echo "Decoding..."
mode='beam'
echo "> Settings: mode=$mode | beam-width=$beam_size | lm-weight=$lm_weight"
echo "  Ensure modeling unit of transducer is the same as that of extra LM."

dec_dir=$dir/${mode}-${beam_size}-$lm_weight
mkdir -p $dec_dir
mkdir -p $dir/enc
for set in $(echo $test_set | tr ':' '\n'); do
    echo "> Decoding: $set"
    python3 ctc-crf/parallel_decode.py \
        --resume=$dir/checks/bestckpt.pt \
        --config=$dir/config.json \
        --input_scp=data/all_ark/$set.scp \
        --output_dir=$dec_dir \
        --enc-out-dir=$dir/enc \
        --spmodel=$spmodel \
        --nj=1 \
        --mode=$mode \
        --beam_size=$beam_size \
        --ext-lm-config=$lmdir/lm_config.json \
        --ext-lm-check=$lmdir/checks/bestckpt.pt \
        --lm-weight=$lm_weight ||
        exit 1

    if [ -f $dec_dir/decode.0-0.tmp ]; then
        cat $dec_dir/decode.?-?.tmp | sort -k 1 >$dec_dir/decode_${set}.txt
        rm $dec_dir/*.tmp
    fi
done

if [ ! $KALDI_ROOT ]; then
    echo "No KALDI_ROOT specified."
    exit 1
fi

for set in $(echo $test_set | tr ':' '\n'); do
    if [ ! -f data/$set/text ]; then
        echo "No true text found: data/$set/text"
        exit 1
    fi
    if [ -f $dec_dir/decode_${set}.txt ]; then
        $KALDI_ROOT/src/bin/compute-wer --text --mode=present ark:data/$set/text ark:$dec_dir/decode_${set}.txt | grep WER >>$dec_dir/result
    else
        echo "No decoded text found: $dec_dir/decode_${set}.txt"
        exit 1
    fi
done

cat $dec_dir/result
