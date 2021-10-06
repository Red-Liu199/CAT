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
trainset=train_sp
devset=dev_sp

if [ ! -d $dir ]; then
    echo "'$dir' is not a directory."
    exit 1
fi

# train sentencepiece
spmdata=data/ai_shell_char
mkdir -p $spmdata

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    bash exp/rnnt-v4/run.sh --sta 1 --sto 1 || exit 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    bash exp/rnnt-v4/run.sh --sta 2 --sto 2 || exit 1
    ln -s $(readlink -f exp/rnnt-v4/pkl) $dir/
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "NN training"

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
        --databalance \
        --amp \
        --grad-norm=5.0 \
        --checkall ||
        exit 1
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    spmodel=$spmdata/spm.model
    # exec/e2edecode.sh $dir "test" $spmodel --cer || exit 1

    # python exec/avgcheckpoint.py --inputs=$dir/checks --num-best 10 || exit 1
    # exec/e2edecode.sh $dir "test" $spmodel --check avg_best_10.pt --cer || exit 1

    python exec/avgcheckpoint.py --inputs $(find $dir/checks/ -name checkpoint.* | sort -g | tail -n 10) \
        --output $dir/checks/avg_last_10.pt || exit 1
    exec/e2edecode.sh $dir "test" $spmodel --check avg_last_10.pt --cer || exit 1
fi
