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
        "--check":{
            "type": "str",
            "default": "bestckpt.pt",
            "help": "Name of checkpoint. Default: bestckpt.pt"
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
        },
        "--cpu":{
            "action":"store_true",
            "help": "Use cpu to decode. Default: False"
        },
        "--cer":{
            "action":"store_true",
            "help": "Compute CER along with WER. Default: False"
        }
    }' $0 $*) && eval $opts || exit 1

echo "Decoding..."
mode='beam'
echo "> Settings: mode=$mode | beam-width=$beam_size | lm-weight=$lm_weight"
echo "  Ensure modeling unit of transducer is the same as that of extra LM."

if [ $cpu == "True" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

dec_dir=$dir/${mode}-${beam_size}-$lm_weight
mkdir -p $dec_dir
mkdir -p $dir/enc
checkpoint=$dir/checks/$check
md5=$(md5sum $checkpoint | cut -d ' ' -f 1)
if [ -f $dir/enc/cinfo.txt ]; then
    if [ ! $(cat $dir/enc/cinfo.txt) ] || [ "$md5" != $(cat $dir/enc/cinfo.txt) ]; then
        rm -rf $dir/enc/*
        echo $md5 >$dir/enc/cinfo.txt
    fi
else
    rm -rf $dir/enc/*
    echo $md5 >$dir/enc/cinfo.txt
fi
unset md5

for set in $(echo $test_set | tr ':' '\n'); do
    echo "> Decoding: $set"
    python3 ctc-crf/parallel_decode.py \
        --resume=$checkpoint \
        --config=$dir/config.json \
        --input_scp=data/all_ark/$set.scp \
        --output_dir=$dec_dir \
        --enc-out-dir=$dir/enc \
        --spmodel=$spmodel \
        --mode=$mode \
        --beam_size=$beam_size \
        --ext-lm-config=$lmdir/lm_config.json \
        --ext-lm-check=$lmdir/checks/bestckpt.pt \
        --lm-weight=$lm_weight ||
        exit 1

    if [ -f $dec_dir/decode.0.tmp ]; then
        cat $dec_dir/decode.*.tmp | sort -k 1 >$dec_dir/decode_${set}.txt
        rm $dec_dir/*.tmp
    fi
done

if [ $check != "bestckpt.pt" ]; then
    echo "Custom checkpoint: $check" >>$dec_dir/result
fi

echo "Use CPU = $cpu" >>$dec_dir/result
for set in $(echo $test_set | tr ':' '\n'); do
    if [ ! -f data/$set/text ]; then
        echo "No true text found: data/$set/text"
        exit 1
    fi
    if [ -f $dec_dir/decode_${set}.txt ]; then
        echo -n "$set " >>$dec_dir/result
        python exec/wer.py data/$set/text $dec_dir/decode_${set}.txt --stripid >>$dec_dir/result || exit 1
        if [ $cer == "True" ]; then
            python exec/wer.py data/$set/text $dec_dir/decode_${set}.txt --stripid --cer >>$dec_dir/result || exit 1
        fi
    else
        echo "No decoded text found: $dec_dir/decode_${set}.txt"
        exit 1
    fi
done

echo "" >>$dec_dir/result
cat $dec_dir/result
