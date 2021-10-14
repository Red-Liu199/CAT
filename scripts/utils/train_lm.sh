# Author: Zheng Huahuan (maxwellzh@outlook.com)
# This script includes the processing of text
# ...and the nn lm training.
# stage 1: convert text into token ids
# stage 2: picklize the datasets
# stage 3: nn lm training

opts=$(python utils/parseopt.py '{
        "spmdata":{
            "type": "str",
            "help": "SentencePiece directory path."
        },
        "--dir":{
            "type": "str",
            "help": "SentencePiece directory path."
        },
        "--stage":{
            "type": "int",
            "default": '3',
            "help": "Start stage. Default: 3"
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

# cp current script to $dir
if [ $dir == "None" ]; then
    dir=$(dirname $0)
else
    cp $0 $dir || exit 1
fi

trainset=
devset=

if [ ! $trainset ] || [ ! $devset ]; then
    echo "Datasets are not properly setup: $trainset | $devset"
    exit 1
fi

textdir=$dir/text
mkdir -p $textdir
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for set in $trainset $devset; do
        src_text=data/$set/text
        # rm seq id to get pure text
        if [ ! -f $src_text ]; then
            echo "No such file $src_text"
            exit 1
        fi
        cat $src_text | cut -d ' ' -f 2- >$textdir/corpus.tmp

        # encode text to token ids
        cat $textdir/corpus.tmp | python3 utils/spm_encode.py --model=$spmdata/spm.model >$textdir/text_id.tmp || exit 1

        # combine seq id with token ids
        cat $src_text | cut -d ' ' -f 1 >$textdir/seq_id.tmp
        paste -d ' ' $textdir/seq_id.tmp $textdir/text_id.tmp >$textdir/${set}.id
        rm $textdir/{seq_id,text_id,corpus}.tmp
    done
    echo "Text to index convertion done."
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    python3 utils/transText2Bin.py --strip $textdir/${devset}.id $textdir/dev.pkl || exit 1

    python3 utils/transText2Bin.py --strip --concat 128 --nj $(nproc) $textdir/${trainset}.id $textdir/tr.pkl || exit 1
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

    python3 rnnt/lm_train.py --seed=0 \
        --world-size 1 --rank 0 -j 1 \
        --batch_size=1152 \
        --dir=$dir \
        --config=$dir/lm_config.json \
        --trset=$textdir/tr.pkl \
        --devset=$textdir/dev.pkl ||
        exit 1
fi
