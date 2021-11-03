# Author: Zheng Huahuan (maxwellzh@outlook.com)
# This script includes the processing of librispeech text
# ...and the nn lm training.
# stage 1: data download, train/dev set split, sentencepiece training
# stage 2: picklize the datasets
# stage 3: nn lm training
set -u
opts=$(python utils/parseopt.py '{
        "--dir":{
            "type": "str",
            "help": "Experiment directory path."
        },
        "--SP":{
            "type": "str",
            "help": "SentencePiece directory path."
        },
        "--stage":{
            "type": "int",
            "default": '3',
            "help": "Start stage. Default: 3"
        },
        "--use-extra":{
            "action": "store_true",
            "help": "Use extra corpus ~40 mil lines. Default: False"
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

if [ $ngpu -ne "-1" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq 0 1 $(($ngpu - 1)) | xargs | tr ' ' ',')
fi
unset ngpu

if [ $dir == "None" ]; then
    dir=$(dirname $0)
fi

############################ DON'T MODIFY CONTENTS ABOVE ############################

# You can manually set GPUs to be used:
# export CUDA_VISIBLE_DEVICES="8,7,6,5,4"

# Setup train/dev/test set here. If there're multiple sets, split them with space
trainset="data/train.txt"
devset="data/valid.txt"
testset="data/test.txt"
SPdir="sentencepiece/"
#####################################################################################

# try resolve sentencepiece directory from outside argument
if [ $SP != "None" ]; then
    export SPdir=$SP
elif [ ! $SPdir ]; then
    echo "You must specify the SentencePiece directory either in the script or with option --SP"
    exit 1
fi
unset SP
python utils/checkfile.py -d $SPdir $dir || exit 1

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Tokenizer initialization"

    # rm seq id to get pure text
    for tr_set in $trainset; do
        cat $tr_set
    done >$SPdir/corpus.tmp || exit 1

    python3 utils/spm_train.py --num_threads=$(nproc) --input=$SPdir/corpus.tmp --model_prefix=$SPdir/spm \
        --bos_id=0 --eos_id=-1 --unk_id=1 --vocab_size=30959 --user_defined_symbols="" \
        --character_coverage=1 --model_type="word" --unk_surface="<unk>" \
        >$SPdir/spm_training.log 2>&1 &&
        echo "SentenPiece training succeed." ||
        {
            echo "Error: check $SPdir/spm_training.log for details"
            exit 1
        }
    rm $SPdir/corpus.tmp

fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo -e "\nStage 1: Text processing"

    for set in $trainset $devset $testset; do
        python utils/checkfile.py -f $set || exit 1
    done

    mkdir -p $dir/text

    textdir=$dir/text
    for prefix in train dev test; do
        dataset=$(eval echo '$'${prefix}set)
        for set in $dataset; do
            cat $set
        done | python3 utils/spm_encode.py --model=$SPdir/spm.model \
            >>$textdir/$prefix.id || exit 1
    done

    # lines of each: 25287 3760 4358
    echo "Stage 1: Convert to token id done."
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo -e "\nStage 2: Data pickling"

    textdir=$dir/text
    python utils/checkfile.py -d $textdir -f $textdir/train.id $textdir/dev.id || exit 1

    mkdir -p $dir/pkl
    # indeed, we won't use the test.pkl, just prepare it for any potential usage.
    python3 utils/transText2Bin.py $textdir/test.id $dir/pkl/test.pkl || exit 1
    python3 utils/transText2Bin.py $textdir/dev.id $dir/pkl/dev.pkl || exit 1
    python3 utils/transText2Bin.py --nj $(nproc) --truncate 128 $textdir/train.id $dir/pkl/tr.pkl || exit 1

fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo -e "\nStage 3: NN training"

    textdir=$dir/pkl
    # parse the number of classes in configuration file
    python3 utils/parseunits.py $SPdir/spm.vocab $dir/config.json || exit 1

    python3 -m cat.lm --seed=0 \
        --world-size 1 --rank 0 -j 1 \
        --dir=$dir \
        --trset=$textdir/tr.pkl \
        --devset=$textdir/dev.pkl \
        --batch_size=512 \
        --grad-norm=5.0 \
        --databalance \
        --checkall \
        --amp ||
        exit 1

    echo -e "\ncommit: \`$(git log -n 1 --pretty=format:"%H")\`" >>$dir/readme.md
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo -e "\nStage 4: Evaluate the test set"

    python3 -m cat.lm \
        --world-size 1 --rank 0 -j 1 \
        --dir=$dir \
        --eval=$textdir/test.pkl ||
        exit 1
fi
