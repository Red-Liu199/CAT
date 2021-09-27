# This script is expected to be executed as
# /bin/bash <path to exp>/run.sh
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
        }
    }' $0 $*) && eval $opts || exit 1

. ./cmd.sh
. ./path.sh

# export CUDA_VISIBLE_DEVICES="8,7,6,5,4"

dir=$(dirname $0)
nj=$(nproc)

if [ ! -d $dir ]; then
    echo "'$dir' is not a directory."
    exit 1
fi

# train sentencepiece
n_units=1024
spmdata=data/spm$n_units
mkdir -p $spmdata

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    # rm seq id to get pure text
    if [ ! -f data/train_tr95_sp/text ]; then
        echo "Training text for sentencepiece not found."
        exit 1
    fi

    cat data/train_tr95_sp/text | cut -d ' ' -f 2- | sed 's/"//g' | sed 's/-//g' | sed 's/,//g' \
        >$spmdata/corpus.tmp

    # sentencepiece training (Here we train in a hack way that set bos_id=0 and unk_id=1.
    # ...Note that bos_id is not used, it will be regarded as <blk> in CTC/RNN-T training.)
    # predefined_syms="<NOISE>"
    predefined_syms=""
    python3 exec/spm_train.py --num_threads=$nj --input=$spmdata/corpus.tmp --model_prefix=$spmdata/spm \
        --bos_id=0 --eos_id=-1 --unk_id=1 --vocab_size=$n_units --user_defined_symbols=$predefined_syms \
        --model_type=unigram \
        >$spmdata/spm_training.log 2>&1 &&
        echo "SentenPiece training succeed." ||
        {
            echo "Error: check $spmdata/spm_training.log for details"
            exit 1
        }
    rm $spmdata/corpus.tmp

    mkdir -p $dir/text
    curdata=$dir/text
    for set in train_tr95_sp train_cv05_sp test_dev93 test_eval92; do
        src_text=data/$set/text
        # rm seq id to get pure text
        if [ ! -f $src_text ]; then
            echo "No such file $src_text"
            exit 1
        fi
        cat $src_text | cut -d ' ' -f 2- | sed 's/"//g' | sed 's/-//g' | sed 's/,//g' \
            >$curdata/corpus.tmp

        # encode text to token ids
        cat $curdata/corpus.tmp | python3 local/spm_encode.py --model=$spmdata/spm.model >$curdata/text_id.tmp

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
    ln -snf $(readlink -f data/all_ark/tr.scp) data/all_ark/train_tr95_sp.scp
    ln -snf $(readlink -f data/all_ark/cv.scp) data/all_ark/train_cv05_sp.scp

    for set in train_tr95_sp; do
        python3 exec/convert_to.py -f=pickle --filer 2000 \
            data/all_ark/$set.scp $dir/text/${set}.id \
            data/$set/weight $dir/pkl/$set.pkl.tmp || exit 1
    done

    for set in train_cv05_sp; do
        python3 exec/convert_to.py -f=pickle \
            data/all_ark/$set.scp $dir/text/${set}.id \
            data/$set/weight $dir/pkl/$set.pkl.tmp || exit 1
    done
    rm -if data/all_ark/train_{tr95,cv05}_sp.scp
    echo "Convert data to pickle done."

    echo "train_tr95_sp -> train"
    python3 local/combinepkl.py -i $dir/pkl/train_tr95_sp.pkl.tmp -o $dir/pkl/tr.pkl || exit 1
    echo "train_cv05_sp -> dev"
    python3 local/combinepkl.py -i $dir/pkl/train_cv05_sp.pkl.tmp -o $dir/pkl/cv.pkl || exit 1

    rm $dir/pkl/*.tmp
    rm -r $dir/text
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "NN training"

    ln -snf $dir/monitor.png link-monitor.png
    python3 ctc-crf/transducer_train.py --seed=0 \
        --world-size 1 \
        --rank=0 \
        --dist-url='tcp://127.0.0.1:13944' \
        --batch_size=256 \
        --dir=$dir \
        --config=$dir/config.json \
        --data=data/ \
        --trset=$dir/pkl/tr.pkl \
        --devset=$dir/pkl/cv.pkl \
        --grad-accum-fold=1 \
        --len-norm="L**1.3" \
        --databalance \
        --checkall \
        --amp \
        --resume=$dir/checks/checkpoint.013.pt ||
        exit 1

fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then

    exec/e2edecode.sh $dir "eval92:dev93" $spmdata/spm.model

fi
