#!/bin/bash
set -u
opts=$(python utils/parseopt.py '{
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
        "--out_prefix":{
            "type": "str",
            "default": "decode",
            "help": "Prefix of decoding output text, which would be <out_prefix>_<test_set>. Default: decode"
        },
        "--check":{
            "type": "str",
            "default": "avg_best_10.pt",
            "help": "Name of checkpoint. Default: avg_best_10.pt"
        },
        "--text_dir":{
            "type": "str",
            "help": "Upper directory of text, assume <text_dir>/<test_set>/text exist. Default: <CAT>/<recipe>/data/"
        },
        "--lm_weight":{
            "type":"float",
            "default": 0.0,
            "help": "External language model weight. Default: 0.0 (disable)"
        },
        "--lmdir":{
            "type": "str",
            "help": "Language model directory. Used at lm_weight > 0. Supposed to include <lmdir>/config.json and <lmdir>/checks/bestckpt.pt"
        },
        "--beam_size":{
            "type":"int",
            "default": 10,
            "help": "Beam search width. Default: 10"
        },
        "--dec-dir":{
            "type":"str",
            "help": "Decode output directory. Default: beam-{beam_size}-{lm_weight}"
        },
        "--algo":{
            "type":"str",
            "default": "lc",
            "choices": ["default", "lc"],
            "help": "Decode algorithm. Default: latency control beam search"
        },
        "--cpu":{
            "action":"store_true",
            "help": "Use cpu to decode. Default: False"
        },
        "--nj":{
            "type":"int",
            "default": -1,
            "help": "Number of threads when using CPU. Default: -1 (all available)."
        },
        "--cer":{
            "action":"store_true",
            "help": "Compute CER along with WER. Default: False"
        }
    }' $0 $*) && eval $opts || exit 1

recipe=$(basename $PWD)
cat_recipe="../../tools/CAT/egs/$recipe/data/"
cat_ark="$cat_recipe/all_ark"
echo "> Settings: algorithm=$algo | beam-width=$beam_size | lm-weight=$lm_weight"

if [ $(echo "$lm_weight > 0.0" | bc -l) -eq 1 ]; then
    prefix="ext_lm=$lm_weight "
    echo "  Ensure modeling unit of transducer is the same as that of extra LM."
else
    prefix=""
fi
if [ -d $spmodel ]; then
    export spmodel=$spmodel/spm.model
    python utils/checkfile.py -f $spmodel || exit 1
fi
if [ $nj -eq "-1" ]; then
    nj=$(nproc)
fi

if [ $cpu == "True" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

if [ $text_dir == "None" ]; then
    text_dir="$cat_recipe"
fi

if [ $dec_dir == "None" ]; then
    dec_dir=$dir/beam-${beam_size}-${lm_weight}
fi
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

test_set=$(echo $test_set | tr ':' ' ')
for set in $test_set; do
    python utils/checkfile.py -f $cat_ark/$set.scp $checkpoint || exit 1

    echo "> Decoding: $set"
    python3 -m cat.rnnt.decode \
        --dist-url="tcp://127.0.0.1:13245" \
        --resume=$checkpoint \
        --config=$dir/config.json \
        --input_scp=$cat_ark/$set.scp \
        --output_dir=$dec_dir \
        --enc-out-dir=$dir/enc \
        --spmodel=$spmodel \
        --algo=$algo \
        --nj=$nj \
        --beam_size=$beam_size \
        --ext-lm-config=$lmdir/config.json \
        --ext-lm-check=$lmdir/checks/bestckpt.pt \
        --lm-weight=$lm_weight ||
        exit 1

    if [ -f $dec_dir/decode.0.tmp ]; then
        cat $dec_dir/decode.*.tmp | sort -k 1 >$dec_dir/${out_prefix}_${set}
        rm $dec_dir/*.tmp
    fi

    if [ -f $dec_dir/nbest.pkl ]; then
        mv $dec_dir/nbest.pkl $dec_dir/${out_prefix}_${set}.nbest.pkl
    fi
done

if [ $check != "bestckpt.pt" ]; then
    echo -n "Custom checkpoint: $check | " >>$dec_dir/result
fi

echo "Use CPU = $cpu" >>$dec_dir/result
for set in $test_set; do
    ground_truth=$text_dir/$set/text
    python utils/checkfile.py -f $ground_truth || exit 1

    if [ -f $dec_dir/${out_prefix}_${set} ]; then
        echo -n "$set $prefix" >>$dec_dir/result
        if [ $cer == "True" ]; then
            python utils/wer.py $ground_truth $dec_dir/${out_prefix}_${set} --stripid --cer >>$dec_dir/result || exit 1
            echo -n "    oracle " >>$dec_dir/result
            python utils/wer.py $ground_truth $dec_dir/${out_prefix}_${set}.nbest.pkl --oracle --cer >>$dec_dir/result || exit 1
        else
            python utils/wer.py $ground_truth $dec_dir/${out_prefix}_${set} --stripid >>$dec_dir/result || exit 1
            echo -n "    oracle " >>$dec_dir/result
            python utils/wer.py $ground_truth $dec_dir/${out_prefix}_${set}.nbest.pkl --oracle >>$dec_dir/result || exit 1
        fi
    else
        echo "No decoded text found: $dec_dir/${out_prefix}_${set}"
        exit 1
    fi
done

echo "" >>$dec_dir/result
cat $dec_dir/result
