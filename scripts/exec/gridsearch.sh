#!/bin/bash

opts=$(python exec/parseopt.py '{
        "dir":{
            "type": "str",
            "help": "Directory of experiment."
        },
        "n_scp":{
            "type": "str",
            "help": "Scp file to load search data. Expand location: data/all_ark/<n_scp>.scp."
        },
        "n_text":{
            "type": "str",
            "help": "Ground truth text file to compute WER. Expand location: data/<n_text>/text."
        },
        "spmodel":{
            "type": "str",
            "help": "Sentence piece model location."
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

loc_scp=data/all_ark/${n_scp}.scp
loc_text=data/$n_text/text

if [ ! -f $loc_scp ]; then
    echo "No such file $loc_scp"
    exit 1
fi

if [ ! -f $loc_text ]; then
    echo "No such file $loc_text"
    exit 1
fi

echo "Settings: mode=$mode | beam-width=$beam_size"
echo "Path: $loc_scp | $loc_text"

dec_dir=$dir/grid-decode-${mode}-${beam_size}
mkdir -p $dec_dir || exit 1
mkdir -p $dir/enc
for weight in $(seq 0.0 0.1 0.3); do
    echo "> Decoding: $weight"
    python3 ctc-crf/parallel_decode.py \
        --resume=$dir/checks/bestckpt.pt \
        --config=$dir/config.json \
        --input_scp=$loc_scp \
        --output_dir=$dec_dir \
        --enc-out-dir=$dir/enc \
        --spmodel=$spmodel \
        --nj=1 \
        --mode=$mode \
        --beam_size=$beam_size \
        --ext-lm-config=$lmdir/lm_config.json \
        --ext-lm-check=$lmdir/checks/bestckpt.pt \
        --lm-weight=$weight ||
        exit 1

    if [ -f $dec_dir/decode.0-0.tmp ]; then
        cat $dec_dir/decode.?-?.tmp | sort -k 1 >$dec_dir/$weight.txt
        rm $dec_dir/*.tmp
    fi

    echo -n "$weight " >>$dec_dir/result
    $KALDI_ROOT/src/bin/compute-wer --text --mode=present ark:$loc_text ark:$dec_dir/$weight.txt | grep WER >>$dec_dir/result
done

cat $dec_dir/result | sed '/^$/d' | sort -g -k 3 | head -n 1 | cut -d ' ' -f 1 >grid_out.tmp
