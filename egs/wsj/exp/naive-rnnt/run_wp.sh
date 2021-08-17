#

. ./cmd.sh
. ./path.sh

data_train=data/train_tr95_sp
data_dev=data/train_cv05_sp

dir="exp/rnnt"
nj=$(nproc)

# train sentencepiece
mkdir -p data/spm
spmdata=data/spm

if [ 1 -lt 0 ]; then
    echo ""
# fi
# rm seq id to get pure text
if [ ! -f $data_train/text ]; then
    echo "No such file $data_train/text"
    exit 1
fi
cat $data_train/text | cut -d ' ' -f 2- > $spmdata/corpus.tmp

# sentencepiece training (Here we train in a hack way that set bos_id=0 and unk_id=1.
# ...Note that bos_id is not used, it will be regarded as <blk> in CTC/RNN-T training.)
predefined_syms="<NOISE>"
python3 ctc-crf/spm_train.py --num_threads=$nj --input=$spmdata/corpus.tmp --model_prefix=$spmdata/spm \
    --bos_id=0 --eos_id=-1 --unk_id=1 --vocab_size=2000 --user_defined_symbols=$predefined_syms \
    --model_type=char   \
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
    cat $tmp_data/text | cut -d ' ' -f 2- > $curdata/corpus.tmp

    # encode text to token ids
    cat $curdata/corpus.tmp | python3 local/spm_encode.py --model=$spmdata/spm.model > $curdata/text_id.tmp

    # combine seq id with token ids
    cat $tmp_data/text | cut -d ' ' -f 1 > $curdata/seq_id.tmp
    paste -d ' ' $curdata/seq_id.tmp $curdata/text_id.tmp > $curdata/text_number
    rm $curdata/{seq_id,text_id,corpus}.tmp
done
echo "Convert to token id done."

# Get lexicon
cat $spmdata/train/text_number | cut -d ' ' -f 2- | tr ' ' '\n' | sort -g | uniq > $spmdata/uniq_id.tmp
cat $spmdata/uniq_id.tmp | python3 local/get_tokenid.py --model=$spmdata/spm.model > $spmdata/tokens.tmp
paste -d ' ' $spmdata/uniq_id.tmp $spmdata/tokens.tmp > $spmdata/tokens.txt
rm $spmdata/{uniq_id,tokens}.tmp

num_outunits=$((1+$(cat data/spm/tokens.txt | tail -n 1 | cut -d ' ' -f 1)))
echo "Number of valid tokens: $(cat data/spm/tokens.txt | wc -l)"
echo "Token ID varies from $(cat data/spm/tokens.txt | head -n 1 | cut -d ' ' -f 1) to $(cat data/spm/tokens.txt | tail -n 1 | cut -d ' ' -f 1)"
echo "You may set the number of output units to $num_outunits"
echo "...and set the <blk>=0)"

# Convert to pickle
python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer 2000 \
    data/all_ark/cv.scp $spmdata/dev/text_number $data_dev/weight $spmdata/cv.pickle \
    > $spmdata/dev/conver2pickle.log 2>&1 || \
    { echo "Error: check $spmdata/dev/conver2pickle.log for details"; exit 1; }
cat $spmdata/dev/conver2pickle.log | tail -n 1

python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer 2000 \
    data/all_ark/tr.scp $spmdata/train/text_number $data_train/weight $spmdata/tr.pickle \
    > $spmdata/train/conver2pickle.log 2>&1 || \
    { echo "Error: check $spmdata/train/conver2pickle.log for details"; exit 1; }
cat $spmdata/train/conver2pickle.log | tail -n 1

echo "Convert data to pickle done."


echo "NN training"

# CUDA_VISIBLE_DEVICES="3"  \
python3 ctc-crf/transducer_train.py --seed=0  \
    --world-size 1 --rank 0  -p 50      \
    --dist-url='tcp://127.0.0.1:13944'  \
    --batch_size=20                     \
    --dir=$dir                          \
    --config=$dir/config.json           \
    --data=data/                        \
    --trset=$spmdata/tr.pickle          \
    --devset=$spmdata/cv.pickle         \
    || exit 1

fi
echo "Decoding..."
mode='prefix'
beam_size=10
dec_dir=$dir/decode-${mode}-${beam_size}
mkdir -p $dec_dir
for set in eval92 dev93; do
    mkdir -p $dec_dir/$set
    python3 ctc-crf/transducer_decode.py    \
        --resume=$dir/ckpt/bestckpt.pt      \
        --config=$dir/config.json           \
        --input_scp=data/all_ark/$set.scp   \
        --output_dir=$dec_dir/$set          \
        --spmodel=$spmdata/spm.model        \
        --mode=$mode                        \
        --beam_size=$beam_size              \
        || exit 1

    if [ -f $dec_dir/$set/decode.0.tmp ]; then
        cat $dec_dir/$set/decode.?.tmp | sort -k 1 > $dec_dir/$set/decode.txt
        rm $dec_dir/$set/*.tmp
    fi
done

echo "" > $dec_dir/result
for set in eval92 dev93; do
    if [ -f $dec_dir/$set/decode.txt ]; then
        $KALDI_ROOT/src/bin/compute-wer --text --mode=present ark:data/test_$set/text ark:$dec_dir/$set/decode.txt >> $dec_dir/result
        echo "" >> $dec_dir/result
    else
        echo "No decoded text found."
    fi
done

cat $dec_dir/result