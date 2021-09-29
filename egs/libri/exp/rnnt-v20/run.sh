# This script is expected to be executed as
opts=$(python exec/parseopt.py '{
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

. ./cmd.sh
. ./path.sh

if [ $ngpu -eq "-1" ]; then
    unset CUDA_VISIBLE_DEVICES
else
    export CUDA_VISIBLE_DEVICES=$(seq 0 1 $(($ngpu - 1)) | xargs | tr ' ' ',')
fi
unset ngpu

# manually set is OK.
# export CUDA_VISIBLE_DEVICES="8,7,6,5,4"

dir=$(dirname $0)
nj=$(nproc)

if [ ! -d $dir ]; then
    echo "'$dir' is not a directory."
    exit 1
fi

# train sentencepiece
n_units=1024
spmdata=data/spm${n_units}_coverall
mkdir -p $spmdata

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    bash exp/rnnt-v19/run.sh --sta 1 --sto 1 || exit 1

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    bash exp/rnnt-v19/run.sh --sta 2 --sto 2 || exit 1
    ln -snf $(readlink -f exp/rnnt-v19/pkl) $dir/pkl
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "NN training"

    ln -snf $dir/monitor.png link-monitor.png
    python3 ctc-crf/transducer_train.py --seed=0 \
        --world-size 1 --rank 0 \
        --dist-url='tcp://127.0.0.1:13944' \
        --batch_size=512 \
        --dir=$dir \
        --config=$dir/config.json \
        --data=data/ \
        --trset=$dir/pkl/tr.pkl \
        --devset=$dir/pkl/cv.pkl \
        --grad-accum-fold=1 \
        --len-norm="L**1.7" \
        --databalance \
        --amp \
        --checkall ||
        exit 1
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    spmodel=$spmdata/spm.model
    exec/e2edecode.sh $dir "test_clean:test_other" $spmodel
fi
