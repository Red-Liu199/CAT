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
n_units=3500
spmdata=data/ai_shell_bpe${n_units}
mkdir -p $spmdata

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    # rm seq id to get pure text
    if [ ! -f data/$trainset/text ]; then
        echo "Training text for sentencepiece not found."
        exit 1
    fi

    cat data/$trainset/text | cut -d ' ' -f 2- >$spmdata/corpus.tmp

    # sentencepiece training (Here we train in a hack way that set bos_id=0 and unk_id=1.
    # ...Note that bos_id is not used, it will be regarded as <blk> in CTC/RNN-T training.)
    # predefined_syms="<NOISE>"
    predefined_syms=""
    python3 exec/spm_train.py --num_threads=$nj --input=$spmdata/corpus.tmp --model_prefix=$spmdata/spm \
        --bos_id=-1 --eos_id=-1 --unk_id=0 --vocab_size=$n_units --user_defined_symbols=$predefined_syms \
        --character_coverage=0.9995 --model_type=unigram --unk_surface="<unk>" \
        >$spmdata/spm_training.log 2>&1 &&
        echo "SentenPiece training succeed." ||
        {
            echo "Error: check $spmdata/spm_training.log for details"
            exit 1
        }
    rm $spmdata/corpus.tmp

    mkdir -p $dir/text
    curdata=$dir/text
    for set in $trainset $devset; do
        src_text=data/$set/text
        # rm seq id to get pure text
        if [ ! -f $src_text ]; then
            echo "No such file $src_text"
            exit 1
        fi
        cat $src_text | cut -d ' ' -f 2- >$curdata/corpus.tmp

        # encode text to token ids
        cat $curdata/corpus.tmp | python3 exec/spm_encode.py --model=$spmdata/spm.model >$curdata/text_id.tmp || exit 1

        # combine seq id with token ids
        cat $src_text | cut -d ' ' -f 1 >$curdata/seq_id.tmp
        paste -d ' ' $curdata/seq_id.tmp $curdata/text_id.tmp >$curdata/${set}.id
        rm $curdata/{seq_id,text_id,corpus}.tmp
    done
    echo "Convert to token id done."
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    # Convert to pickle
    mkdir -p $dir/pkl
    ln -snf $(readlink -f data/all_ark/tr.scp) data/all_ark/$trainset.scp
    ln -snf $(readlink -f data/all_ark/cv.scp) data/all_ark/$devset.scp

    for set in $trainset; do
        python3 exec/convert_to.py -f=pickle --filer 2000 \
            data/all_ark/$set.scp $dir/text/${set}.id \
            data/$set/weight $dir/pkl/$set.pkl.tmp || exit 1
    done

    for set in $devset; do
        python3 exec/convert_to.py -f=pickle \
            data/all_ark/$set.scp $dir/text/${set}.id \
            data/$set/weight $dir/pkl/$set.pkl.tmp || exit 1
    done
    rm -if data/all_ark/{$trainset, $devset}.scp
    echo "Convert data to pickle done."

    echo "$trainset -> tr"
    python3 exec/combinepkl.py -i $dir/pkl/$trainset.pkl.tmp -o $dir/pkl/tr.pkl || exit 1
    echo "$devset -> cv"
    python3 exec/combinepkl.py -i $dir/pkl/$devset.pkl.tmp -o $dir/pkl/cv.pkl || exit 1

    rm $dir/pkl/*.tmp
    rm -r $dir/text
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "NN training"

    python3 ctc-crf/transducer_train.py --seed=0 \
        --world-size 1 --rank 0 \
        --dist-url='tcp://127.0.0.1:13944' \
        --batch_size=512 \
        --dir=$dir \
        --config=$dir/config.json \
        --trset=$dir/pkl/tr.pkl \
        --devset=$dir/pkl/cv.pkl \
        --grad-accum-fold=1 \
        --databalance \
        --amp --resume=$dir/checks/checkpoint.018.pt \
        --checkall ||
        exit 1
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    spmodel=$spmdata/spm.model
    exec/e2edecode.sh $dir "test" $spmodel || exit 1

    python exec/avgcheckpoint.py --inputs=$dir/checks --num-best 10 || exit 1
    exec/e2edecode.sh $dir "test" $spmodel --check avg_best_10.pt

fi