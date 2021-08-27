# This script is expected to be executed as 
# /bin/bash <path to exp>/run.sh

. ./cmd.sh
. ./path.sh

data_train="data/train_tr95"
data_dev="data/train_cv05"
stage=3
stop_stage=3
export CUDA_VISIBLE_DEVICES=8 #"8,7,6,5,4"

dir=$(dirname $0)
nj=$(nproc)

if [ ! -d $dir ]; then
    echo "'$dir' is not a directory."
    exit 1
fi

# train sentencepiece
mkdir -p data/spm
spmdata=data/spm

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    # rm seq id to get pure text
    if [ ! -f $data_train/text ]; then
        echo "No such file $data_train/text"
        exit 1
    fi
    cat $data_train/text | cut -d ' ' -f 2- | sed 's/"//g' | sed 's/-//g' | sed 's/,//g' \
        > $spmdata/corpus.tmp

    # sentencepiece training (Here we train in a hack way that set bos_id=0 and unk_id=1.
    # ...Note that bos_id is not used, it will be regarded as <blk> in CTC/RNN-T training.)
    # predefined_syms="<NOISE>"
    predefined_syms=""
    python3 ctc-crf/spm_train.py --num_threads=$nj --input=$spmdata/corpus.tmp --model_prefix=$spmdata/spm \
        --bos_id=0 --eos_id=-1 --unk_id=1 --vocab_size=1024 --user_defined_symbols=$predefined_syms \
        --model_type=unigram   \
        > $spmdata/spm_training.log 2>&1 \
        && echo "SentenPiece training succeed." || \
        { echo "Error: check $spmdata/spm_training.log for details"; exit 1; }
    rm $spmdata/corpus.tmp

    for set in train dev; do
        mkdir -p $spmdata/$set
        curdata=$spmdata/$set

        tmp_data=`eval echo '$'data_$set`
        # rm seq id to get pure text
        if [ ! -f $tmp_data/text ]; then
            echo "No such file $tmp_data/text"
            exit 1
        fi
        cat $tmp_data/text | cut -d ' ' -f 2- | sed 's/"//g' | sed 's/-//g' | sed 's/,//g' \
            > $curdata/corpus.tmp

        # encode text to token ids
        cat $curdata/corpus.tmp | python3 local/spm_encode.py --model=$spmdata/spm.model > $curdata/text_id.tmp

        # combine seq id with token ids
        cat $tmp_data/text | cut -d ' ' -f 1 > $curdata/seq_id.tmp
        paste -d ' ' $curdata/seq_id.tmp $curdata/text_id.tmp > $curdata/text_number
        rm $curdata/{seq_id,text_id,corpus}.tmp
    done
    echo "Convert to token id done."

    # Get lexicon
    cat $spmdata/train/text_number | cut -d ' ' -f 2- | tr ' ' '\n' | awk '{ cnts[$0] += 1 } END { for (v in cnts) print v }' \
        | sort -g > $spmdata/uniq_id.tmp
    cat $spmdata/uniq_id.tmp | python3 local/get_tokenid.py --model=$spmdata/spm.model > $spmdata/tokens.tmp
    paste -d ' ' $spmdata/uniq_id.tmp $spmdata/tokens.tmp > $spmdata/tokens.txt
    rm $spmdata/{uniq_id,tokens}.tmp

    num_outunits=$((1+$(cat data/spm/tokens.txt | tail -n 1 | cut -d ' ' -f 1)))
    echo "Number of valid tokens: $(cat data/spm/tokens.txt | wc -l)"
    echo "Token ID varies from $(cat data/spm/tokens.txt | head -n 1 | cut -d ' ' -f 1) to $(cat data/spm/tokens.txt | tail -n 1 | cut -d ' ' -f 1)"
    echo "You may set the number of output units to $num_outunits"
    echo "...and set the <blk>=0)"
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    # Convert to pickle
    python3 ctc-crf/convert_to.py -f=pickle --filer 2000 \
        data/all_ark/cv.scp $spmdata/dev/text_number $data_dev/weight $spmdata/cv.pickle \
        > $spmdata/dev/conver2pickle.log 2>&1 || \
        { echo "Error: check $spmdata/dev/conver2pickle.log for details"; exit 1; }
    cat $spmdata/dev/conver2pickle.log | tail -n 1

    python3 ctc-crf/convert_to.py -f=pickle --filer 2000 \
        data/all_ark/tr.scp $spmdata/train/text_number $data_train/weight $spmdata/tr.pickle \
        > $spmdata/train/conver2pickle.log 2>&1 || \
        { echo "Error: check $spmdata/train/conver2pickle.log for details"; exit 1; }
    cat $spmdata/train/conver2pickle.log | tail -n 1

    echo "Convert data to pickle done."
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "NN training"

    ln -snf $dir/ckpt/monitor.png link-monitor.png
    python3 ctc-crf/transducer_train.py --seed=0  \
        --world-size 1 --rank 0             \
        --dist-url='tcp://127.0.0.1:13944'  \
        --batch_size=40                     \
        --dir=$dir                          \
        --config=$dir/config.json           \
        --data=data/                        \
        --trset=$spmdata/tr.pickle          \
        --devset=$spmdata/cv.pickle         \
        --grad-accum-fold=15                \
        || exit 1

    ./decode.sh $dir
fi
