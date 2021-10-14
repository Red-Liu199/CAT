#!/bin/bash

opts=$(python utils/parseopt.py '{
        "dir":{
            "type": "str",
            "help": "Directory of experiment."
        },
        "spmodel":{
            "type": "str",
            "help": "Sentence piece model location."
        },
        "n_scp":{
            "type": "str",
            "help": "Scp file to load search data. Expand location: data/all_ark/<n_scp>.scp."
        },
        "n_text":{
            "type": "str",
            "help": "Ground truth text file to compute WER. Expand location: data/<n_text>/text."
        },
        "lmdir":{
            "type": "str",
            "help": "Language model directory. Supposed to include <lmdir>/lm_config.json and <lmdir>/checks/bestckpt.pt"
        },
        "--beam_size":{
            "type":"int",
            "default": 5,
            "help": "Beam search width. Default: 5"
        },
        "--check":{
            "type": "str",
            "default": "bestckpt.pt",
            "help": "Name of checkpoint. Default: bestckpt.pt"
        },
        "--cer":{
            "action":"store_true",
            "help": "Compute CER along with WER. Default: False"
        }
    }' $0 $*) && eval $opts || exit 1

echo "Grid Searching..."
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

dec_dir=$dir/grid-${mode}-${beam_size}

for weight in $(seq 0.0 0.1 0.5); do
    echo "> Decoding: $weight"
    if [ -f $dec_dir/${n_scp}_${weight} ]; then
        continue
    fi
    if [ $cer == "True" ]; then
        ./utils/e2edecode.sh $dir $n_scp $spmodel --check=$check --lm_weight=$weight \
            --lmdir=$lmdir --beam_size=$beam_size --dec-dir=$dec_dir --cer || exit 1
    else
        ./utils/e2edecode.sh $dir $n_scp $spmodel --check=$check --lm_weight=$weight \
            --lmdir=$lmdir --beam_size=$beam_size --dec-dir=$dec_dir || exit 1
    fi
    mv $dec_dir/decode_${n_scp}.txt $dec_dir/${n_scp}_${weight} || exit 1
done

if [ $cer == "True" ]; then
    cat $dec_dir/result | grep "CER" | sort -g -k 4 >$dec_dir/grid_out
else
    cat $dec_dir/result | grep "WER" | sort -g -k 4 >$dec_dir/grid_out
fi
