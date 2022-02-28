set -u
set -e

usecpu=True
lm_dir=$1
checkpoint="avg_60_10.pt"
eval_sets="dev_other"
tokenizer=$(cat $lm_dir/hyper-p.json | python -c "import json,sys; print(json.load(sys.stdin)['tokenizer']['location'])")
fmt_am_nbest="exp/rnnt-v27/decode/rnnt-256_algo-rna_nolm_best_10_%s.nbest"

fmt_lm="$lm_dir/rescore/$(echo $fmt_am_nbest | cut -d '/' -f 2)"
mkdir -p $fmt_lm
fmt_beta="$fmt_lm/$(basename $fmt_am_nbest).beta"
fmt_lm="$fmt_lm/$(basename $fmt_am_nbest).lm"

[ ! -f $tokenizer ] && echo "No such tokenizer file found: $tokenizer" && exit 1

for subset in $eval_sets; do
    echo $(printf $fmt_beta $subset)
    [ ! -f $(printf $fmt_beta $subset) ] &&
        python utils/interpolate_nbests.py $(printf $fmt_beta $subset) \
            --nbestlist $(printf $fmt_am_nbest $subset) --weight 0.0 \
            --ins-penalty 1.0 --tokenizer $tokenizer &
done
wait

if [ $usecpu == "True" ]; then
    for subset in $eval_sets; do
        echo $(printf $fmt_lm $subset)
        cache_file="/tmp/$(basename $fmt_am_nbest).tmp"
        [ ! -f $(printf $fmt_lm $subset) ] &&
            python -m cat.lm.rescore --nj 10 --cpu $(printf $fmt_am_nbest $subset) /dev/null \
                --config $lm_dir/config.json --resume $lm_dir/checks/$checkpoint \
                --tokenizer $tokenizer --save-lm-nbest $(printf $fmt_lm $subset) &
    done
    wait
else
    for subset in $eval_sets; do
        echo $(printf $fmt_lm $subset)
        [ ! -f $(printf $fmt_lm $subset) ] &&
            python -m cat.lm.rescore \
                $(printf $fmt_am_nbest $subset) /dev/null \
                --config $lm_dir/config.json --resume $lm_dir/checks/$checkpoint \
                --tokenizer $tokenizer --save-lm-nbest $(printf $fmt_lm $subset)
    done
    wait
fi

for subset in $eval_sets; do
    python utils/lmweight_search.py \
        --nbestlist $(printf $fmt_am_nbest $subset) \
        $(printf $fmt_lm $subset) $(printf $fmt_beta $subset) \
        --search 0 1 1 \
        --ground-truth "/mnt/workspace/zhenghh/CAT/egs/$(basename $(pwd))/data/$subset/text" &
done | grep SER | grep -v tun
wait
