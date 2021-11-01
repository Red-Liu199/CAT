# This script is expected to be executed as
# /bin/bash <path to exp>/run.sh
opts=$(python utils/parseopt.py '{
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

if [ $ngpu -ne "-1" ]; then
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
fi
############################ DON'T MODIFY CONTENTS ABOVE ############################

# Setup train/dev/test set here. If there're multiple sets, split them with space
trainset="train_960"
devset="dev_clean dev_other"
testset="test_clean test_other"

########## Train sentencepiece ##########
char=false
n_units=1024
#########################################
if [ $char == "true" ]; then
    bpemode=char
    n_units=100000
    SPdir=sentencepiece/${recipe}_char
else
    bpemode=unigram
    SPdir=sentencepiece/${recipe}_${n_units}
fi
mkdir -p $SPdir

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    bash exp/rnnt-v25/run.sh --sta 1 --sto 1 --dir=$dir || exit 1

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    bash exp/template/run_rnnt.sh --sta 2 --sto 2 --dir=$dir --SP=$SPdir || exit 1

fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "NN training"

    # parse the number of classes in configuration file
    python3 utils/parseunits.py $SPdir/spm.vocab $dir/config.json || exit 1

    python3 -m cat.rnnt --seed=0 \
        --world-size 1 --rank=0 \
        --batch_size=128 \
        --dir=$dir \
        --trset=$dir/pkl/tr.pkl \
        --devset=$dir/pkl/cv.pkl \
        --grad-accum-fold=16 \
        --databalance \
        --checkall \
        --amp ||
        exit 1

    echo -e "\ncommit: \`$(git log -n 1 --pretty=format:"%H")\`" >>$dir/readme.md
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then

    bash exp/template/run_rnnt.sh --sta 4 --sto 4 --dir=$dir --SP=$SPdir || exit 1

fi
