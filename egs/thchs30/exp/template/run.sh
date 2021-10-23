# This script is expected to be executed as
# /bin/bash <path to exp>/run.sh
set -u
opts=$(python utils/parseopt.py '{
        "--stage":{
            "type": "int",
            "default": '3',
            "help": "Start stage. Default: 3"
        },
        "--dir":{
            "type": "str",
            "help": "Experiment directory path."
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
        },
        "--SP":{
            "type": "str",
            "help": "Optional: SentencePiece directory path. Default: use the path in the script."
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

if [ $dir == "None" ]; then
    dir=$(dirname $0)
fi
python utils/checkfile.py -d $dir $cat_recipe || exit 1

############################ DON'T MODIFY CONTENTS ABOVE ############################
# Setup train/dev/test set here. If there're multiple sets, split them with space
trainset="train_sp"
devset="dev_sp"
testset="test"

########## Train sentencepiece ##########
char=true
n_units=5000
#########################################
if [ $SP == "None" ]; then
    if [ $char == "true" ]; then
        bpemode=char
        n_units=100000
        SPdir=sentencepiece/${recipe}_char
    else
        bpemode=unigram
        SPdir=sentencepiece/${recipe}_${n_units}
    fi
else
    # overwrite settings
    export SPdir=$SP
    unset SP
fi
mkdir -p $SPdir

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for set in $trainset $devset $testset; do
        python utils/checkfile.py -f ${cat_recipe}/${set}/text || exit 1
    done
    # rm seq id to get pure text
    for tr_set in $trainset; do
        cat ${cat_recipe}/$tr_set/text
    done | cut -d ' ' -f 2- | sed 's/ //g' >$SPdir/corpus.tmp || exit 1

    python3 utils/spm_train.py --num_threads=$(nproc) --input=$SPdir/corpus.tmp --model_prefix=$SPdir/spm \
        --bos_id=0 --eos_id=-1 --unk_id=1 --vocab_size=$n_units --user_defined_symbols="" \
        --character_coverage=1 --model_type=$bpemode --unk_surface="<unk>" --add_dummy_prefix=False \
        >$SPdir/spm_training.log 2>&1 &&
        echo "SentenPiece training succeed." ||
        {
            echo "Error: check $SPdir/spm_training.log for details"
            exit 1
        }
    rm $SPdir/corpus.tmp

    mkdir -p $dir/text
    curdata=$dir/text
    for set in $trainset $devset; do
        src_text=${cat_recipe}/${set}/text
        # rm seq id to get pure text
        cat $src_text | cut -d ' ' -f 2- | sed 's/ //g' >$curdata/corpus.tmp

        # encode text to token ids
        cat $curdata/corpus.tmp | python3 utils/spm_encode.py --model=$SPdir/spm.model >$curdata/text_id.tmp

        # combine seq id with token ids
        cat $src_text | cut -d ' ' -f 1 >$curdata/seq_id.tmp
        paste -d ' ' $curdata/seq_id.tmp $curdata/text_id.tmp >$curdata/${set}.id
        rm $curdata/{seq_id,text_id,corpus}.tmp
    done
    echo "Convert to token id done."

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for set in $trainset $devset $testset; do
        python utils/checkfile.py -f ${cat_recipe}/all_ark/${set}.scp || exit 1
    done
    # Convert to pickle
    mkdir -p $dir/pkl

    for set in $trainset; do
        python3 utils/convert_to.py -f=pickle --filer 2000 \
            ${cat_recipe}/all_ark/$set.scp $dir/text/${set}.id \
            $dir/pkl/$set.pkl.tmp || exit 1
    done

    for set in $devset; do
        python3 utils/convert_to.py -f=pickle \
            ${cat_recipe}/all_ark/$set.scp $dir/text/${set}.id \
            $dir/pkl/$set.pkl.tmp || exit 1
    done
    echo "Convert data to pickle done."

    echo "$trainset -> train"
    python3 utils/combinepkl.py -i $(echo $trainset | tr ' ' '\n' | sed "s|^|$dir/pkl/|g" | sed 's|$|\.pkl\.tmp|g') \
        -o $dir/pkl/tr.pkl || exit 1

    echo "$devset -> dev"
    python3 utils/combinepkl.py -i $(echo $devset | tr ' ' '\n' | sed "s|^|$dir/pkl/|g" | sed 's|$|\.pkl\.tmp|g') \
        -o $dir/pkl/cv.pkl || exit 1

    rm $dir/pkl/*.tmp
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "NN training"

    # parse the number of classes in configuration file
    python3 utils/parseunits.py $SPdir/spm.vocab $dir/config.json || exit 1

    python3 rnnt/transducer_train.py --seed=0 \
        --world-size 1 --rank=0 \
        --batch_size=256 \
        --dir=$dir \
        --config=$dir/config.json \
        --trset=$dir/pkl/tr.pkl \
        --devset=$dir/pkl/cv.pkl \
        --grad-accum-fold=1 \
        --databalance \
        --checkall \
        --amp ||
        exit 1

    echo -e "\ncommit: \`$(git log -n 1 --pretty=format:"%H")\`" >>$dir/readme.md
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    SPmodel=$SPdir/spm.model
    python utils/checkfile.py -f $SPmodel -d $dir || exit 1

    # generate averaging models
    if [ ! -f $dir/checks/avg_best_10.pt ]; then
        python utils/avgcheckpoint.py --inputs=$dir/checks --num-best 10 || exit 1
    fi
    if [ ! -f $dir/checks/avg_last_10.pt ]; then
        python utils/avgcheckpoint.py --inputs $(find $dir/checks/ -name checkpoint.* | sort -g | tail -n 10) \
            --output $dir/checks/avg_last_10.pt || exit 1
    fi

    for checkpoint in bestckpt.pt avg_last_10.pt avg_best_10.pt; do
        utils/e2edecode.sh $dir $(echo $testset | tr ' ' ':') $SPmodel \
            --out_prefix=$(echo $checkpoint | cut -d '.' -f 1) \
            --check=$checkpoint --cer || exit 1
    done
fi
