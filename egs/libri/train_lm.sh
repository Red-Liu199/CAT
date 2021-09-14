# Author: Zheng Huahuan (maxwellzh@outlook.com)
# This script includes the processing of librispeech extra text
# ...and the nn lm training.
# stage 1: data download, train/dev set split, sentencepiece training
# stage 2: picklize the datasets
# stage 3: nn lm training

stage=3
stop_stage=3

text_dir=data/text
mkdir -p $text_dir

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    text=$text_dir/librispeech-lm-norm.txt.gz
    if [ ! -f $text_dir/librispeech.txt ]; then
        if [ ! -f $text ]; then
            wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P $text_dir
        fi

        echo -n >$text_dir/dev.txt
        # hold out one in every 2000 lines as dev data.
        gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%2000 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/librispeech.txt
    fi


    spmdata=$text_dir/spm
    mkdir -p $spmdata
    if [ ! -f $spmdata/spm.model ]; then
        # sentencepiece training (Here we train in a hack way that set bos_id=0 and unk_id=1.
        # ...Note that bos_id is not used, it will be regarded as <blk> in CTC/RNN-T training.)
        # predefined_syms="<NOISE>"
        predefined_syms=""
        python3 ctc-crf/spm_train.py --num_threads=20 --input=$text_dir/librispeech.txt --model_prefix=$spmdata/spm \
            --bos_id=0 --eos_id=-1 --unk_id=1 --vocab_size=1024 --user_defined_symbols=$predefined_syms \
            --model_type=unigram  --input_sentence_size=1000000 --shuffle_input_sentence=true \
            > $spmdata/spm_training.log 2>&1 \
            && echo "SentenPiece training succeed." || \
            { echo "Error: check $spmdata/spm_training.log for details"; exit 1; }
    fi

    mkdir -p $text_dir/data
    if [ ! -f $text_dir/data/tr.tmp ]; then
        rsync -avPh $text_dir/librispeech.txt $text_dir/data/tr.tmp
    fi
    if [ ! -f $text_dir/data/dev.tmp ]; then
        rsync -avPh $text_dir/dev.txt $text_dir/data/dev.tmp
    fi

    for set in dev tr; do
        processing_text=$text_dir/data/$set.tmp
        # encode text to token ids
        cat $processing_text | python3 local/spm_encode.py --model=$spmdata/spm.model > $text_dir/data/${set}.id
        rm $set.tmp
    done
    echo "Convert to token id done."

    # Get lexicon
    awk 'NF' $text_dir/data/tr.id | tr ' ' '\n' | awk '{ cnts[$0] += 1 } END { for (v in cnts) print v }'   \
        | sort -g > $text_dir/uniq_id.tmp
    cat $text_dir/uniq_id.tmp | python3 local/get_tokenid.py --model=$spmdata/spm.model > $text_dir/tokens.tmp
    paste -d ' ' $text_dir/uniq_id.tmp $text_dir/tokens.tmp > $text_dir/tokens.txt
    rm $text_dir/{uniq_id,tokens}.tmp

    num_outunits=$((1+$(cat $text_dir/tokens.txt | tail -n 1 | cut -d ' ' -f 1)))
    echo "Number of valid tokens: $(cat $text_dir/tokens.txt | wc -l)"
    echo "Token ID varies from $(cat $text_dir/tokens.txt | head -n 1 | cut -d ' ' -f 1) to $(cat data/spm/tokens.txt | tail -n 1 | cut -d ' ' -f 1)"
    echo "You may set the number of output units to $num_outunits"
    echo "...and set the <blk>=0)"
fi


# if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
#     if [ ! -f $text_dir/data/dev.pkl ]; then
#         python3 local/transText2Bin.py --concat=256 $text_dir/data/dev.id $text_dir/data/dev.pkl || exit 1
#     fi
#     if [ ! -f $text_dir/data/tr.pkl ]; then
#         python3 local/transText2Bin.py --nj $(nproc) --concat=256 $text_dir/data/tr.id $text_dir/data/tr.pkl || exit 1
#     fi
# fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    mkdir -p data/spm/data

    python3 local/transText2Bin.py --strip data/spm/dev/text_number data/spm/data/dev.pkl || exit 1

    python3 local/transText2Bin.py --strip --nj $(nproc) data/spm/train/text_number data/spm/data/tr.pkl || exit 1
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    dir=$1

    if [ ! $dir ] || [ ! -d $dir ]; then
        echo "No such directory $dir"
        exit 1
    fi

    if [ -f $dir/train_lm.sh ]; then
        if [ $dir != "$(dirname $0)" ] && [ $dir != "$(dirname $0)/" ]; then
            echo "Found $dir/train_lm.sh, ensure it's useless, rm it then re-run this script."
            exit 0
        fi
    else
        cp $0 $dir
    fi

    CUDA_VISIBLE_DEVICES="8,7,6,5,4"  \
    python3 ctc-crf/lm_train.py --seed=0        \
        --world-size 1 --rank 0 -j 1            \
        --batch_size=1280                       \
        --dir=$dir                              \
        --config=$dir/lm_config.json            \
        --data=data/                            \
        --trset=data/spm/data/tr.pkl            \
        --devset=data/spm/data/dev.pkl          \
        --grad-accum-fold=1                     \
        || exit 1
fi