# Author: Zheng Huahuan (maxwellzh@outlook.com)
# This script includes the processing of librispeech extra text
# ...and the nn lm training.
# stage 1: data download, train/dev set split, sentencepiece training
# stage 2: picklize the datasets
# stage 3: nn lm training
set -u
opts=$(python utils/parseopt.py '{
        "SPdir":{
            "type": "str",
            "help": "SentencePiece directory path."
        },
        "--stage":{
            "type": "int",
            "default": '3',
            "help": "Start stage. Default: 3"
        },
        "--dir":{
            "type": "str",
            "help": "SentencePiece directory path."
        },
        "--stop_stage":{
            "type": "int",
            "default": 100,
            "help": "Stop stage. Default: 100"
        },
        "--ngpu":{
            "type": "int",
            "default": -1,
            "help": "Number of GPUs to used. Default: -1 (all available GPUs)"
        }
    }' $0 $*) && eval $opts || exit 1

if [ $ngpu -eq "-1" ]; then
    unset CUDA_VISIBLE_DEVICES
else
    export CUDA_VISIBLE_DEVICES=$(seq 0 1 $(($ngpu - 1)) | xargs | tr ' ' ',')
fi
unset ngpu

# manually set is OK.
# export CUDA_VISIBLE_DEVICES="8,7,6,5,4"

recipe=$(basename $PWD)
cat_recipe="../../tools/CAT/egs/$recipe/data"
# cp current script to $dir
if [ $dir == "None" ]; then
    dir=$(dirname $0)
else
    python utils/checkfile.py -d $dir || exit 1
    cp $0 $dir || exit 1
fi
############################ DON'T MODIFY CONTENTS ABOVE ############################

# Setup train/dev/test set here. If there're multiple sets, split them with space
trainset="train_960"
devset="dev_clean dev_other"
testset="test_clean test_other"

textdir=data/corpus
mkdir -p $textdir
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    if [ ! -f $textdir/librispeech.txt ]; then
        if [ ! -f $textdir/librispeech-lm-norm.txt.gz ]; then
            wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P $textdir || exit 1
        fi

        gunzip -c $text >$textdir/librispeech.txt || exit 1
        rm $textdir/librispeech-lm-norm.txt.gz
        echo "Fetched text corpus. At $textdir/librispeech.txt"
    fi

    mkdir -p $dir/text
    for tr_set in $trainset; do
        cat ${cat_recipe}/$tr_set/text
    done | cut -d ' ' -f 2- >$dir/text/extra_tr.tmp || exit 1

    # split large corpus by 1e6 lines. ~40,000,000 lines in total
    cat {$textdir/librispeech.txt,$dir/text/extra_tr.tmp} | split -d -l $((40699502 / $(nproc))) - tmp_corpus_ || exit 1
    for i in $(ls tmp_corpus_*); do
        cat $i | python3 utils/spm_encode.py --model=$SPdir/spm.model &
    done >$dir/text/train.id || exit 1
    wait

    for set in $devset; do
        cat ${cat_recipe}/$set/text
    done | cut -d ' ' -f 2- |
        python3 utils/spm_encode.py --model=$SPdir/spm.model \
            >$dir/text/dev.id || exit 1

    rm $dir/text/extra_tr.tmp
    rm tmp_corpus_*
    echo "Convert to token id done."
fi

textdir=$dir/text
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    python utils/checkfile.py -d $textdir -f $textdir/train.id $textdir/dev.id || exit 1

    mkdir -p $dir/pkl
    python3 utils/transText2Bin.py $textdir/dev.id $dir/pkl/dev.pkl || exit 1
    python3 utils/transText2Bin.py --nj $(nproc) $textdir/train.id $dir/pkl/tr.pkl || exit 1

fi

textdir=$dir/pkl
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

    # parse the number of classes in configuration file
    python3 utils/parseunits.py $SPdir/spm.vocab $dir/config.json || exit 1

    python3 rnnt/lm_train.py --seed=0 \
        --world-size 1 --rank 0 -j 1 \
        --batch_size=1024 \
        --dir=$dir \
        --config=$dir/config.json \
        --trset=$textdir/tr.pkl \
        --devset=$textdir/dev.pkl ||
        exit 1

    echo -e "\ncommit: \`$(git log -n 1 --pretty=format:"%H")\`" >>$dir/readme.md
fi
