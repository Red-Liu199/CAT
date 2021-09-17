#!/bin/bash

dir=$1
if [ ! $dir ]; then
    echo "You need to specify the experiment directory."
    exit 1
fi

. ./path.sh

echo "Decoding..."
mode='beam'
beam_size=5
lmdir="exp/lm1024"
spmdata="data/spm1024"

if [ ! $lmdir ]; then
    echo "You need to specify the extra lm directory."
    exit 1
elif [ ! -d $lmdir ]; then
    echo "No such directory $lmdir"
    exit 1
fi

n_scp=test_other
n_text=test_other

loc_scp=data/all_ark/${n_scp}.scp
loc_text=data/$n_text/text

if [ ! -f $loc_scp ];then
    echo "No such file $loc_scp"
    exit 1
fi

if [ ! -f $loc_text ];then
    echo "No such file $loc_text"
    exit 1
fi

echo "Settings: mode=$mode | beam-width=$beam_size"
echo "Path: $loc_scp | $loc_text"

dec_dir=$dir/grid-decode-${mode}-${beam_size}
mkdir -p $dec_dir || exit 1
mkdir -p $dir/enc
echo >> $dec_dir/result
for weight in $(seq 0.1 0.1 0.5) ; do
    echo "> Decoding: $weight"
    python3 ctc-crf/parallel_decode.py      \
        --resume=$dir/checks/bestckpt.pt    \
        --config=$dir/config.json           \
        --input_scp=$loc_scp                \
        --output_dir=$dec_dir               \
        --enc-out-dir=$dir/enc              \
        --spmodel=$spmdata/spm.model        \
        --nj=1                              \
        --mode=$mode                        \
        --beam_size=$beam_size              \
        --ext-lm-config=$lmdir/lm_config.json \
        --ext-lm-check=$lmdir/checks/bestckpt.pt   \
        --lm-weight=$weight \
        || exit 1

    if [ -f $dec_dir/decode.0-0.tmp ]; then
        cat $dec_dir/decode.?-?.tmp | sort -k 1 > $dec_dir/$weight.txt
        rm $dec_dir/*.tmp
    fi
    
    echo $weight >> $dec_dir/result
    $KALDI_ROOT/src/bin/compute-wer --text --mode=present ark:$loc_text ark:$dec_dir/$weight.txt | grep WER >> $dec_dir/result
done

cat $dec_dir/result
