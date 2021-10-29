# Author: Zheng Huahuan (maxwellzh@outlook.com)
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

recipe=$(basename $PWD)
cat_recipe="../../tools/CAT/egs/$recipe/data"

if [ $dir == "None" ]; then
    dir=$(dirname $0)
fi

############################ DON'T MODIFY CONTENTS ABOVE ############################

# You can manually set GPUs to be used:
# export CUDA_VISIBLE_DEVICES="8,7,6,5,4"

# Setup train/dev/test set here. If there're multiple sets, split them with space
trainset="train_tr95"
devset="train_cv05"
testset="test_eval92 test_dev93"
SPdir=
#####################################################################################

# try resolve sentencepiece directory from outside argument
if [ $SP != "None" ]; then
    export SPdir=$SP
elif [ ! $SPdir ]; then
    echo "You must specify the SentencePiece directory either in the script or with option --SP"
    exit 1
fi
unset SP
python utils/checkfile.py -d $SPdir $dir $cat_recipe || exit 1

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Text processing"

    for set in $trainset $devset $testset; do
        python utils/checkfile.py -f ${cat_recipe}/$set/text || exit 1
    done

    textdir=$dir/text
    mkdir -p $textdir
    if [ -d $dir/text ] && [ $(ls $dir/text | wc -l) -gt 0 ]; then
        mkdir -p $dir/.backup
        mv $dir/text $dir/.backup/ || exit 1
        mkdir -p $dir/text
        echo "Stage 1: move existing text files into $dir/.backup/"
    fi

    for prefix in train dev test; do
        dataset=$(eval echo '$'${prefix}set)
        for set in $dataset; do
            cat ${cat_recipe}/$set/text
        done | cut -d ' ' -f 2- |
            python3 utils/spm_encode.py --model=$SPdir/spm.model \
                >>$textdir/$prefix.id || exit 1
    done

    # lines of each: 35508 1908 836
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
    python3 utils/transText2Bin.py --nj $(nproc) --truncate 64 $textdir/train.id $dir/pkl/tr.pkl || exit 1

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
        --batch_size=2048 \
        --grad-norm=5.0 \
        --databalance \
        --checkall \
        --amp ||
        exit 1

    echo -e "\ncommit: \`$(git log -n 1 --pretty=format:"%H")\`" >>$dir/readme.md
fi
