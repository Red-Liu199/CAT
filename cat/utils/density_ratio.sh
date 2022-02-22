set -u
set -e

lm_src=$1
lm_dst=$2

fmt_am_nbest="exp/rnnt-v27/decode/rnnt-128_algo-rna_nolm_best_10_%s.nbest"

fmt_lm="rescore/$(echo $fmt_am_nbest | cut -d '/' -f 2)/$(basename $fmt_am_nbest).lm"
fmt_lm_src="$lm_src/$fmt_lm"
fmt_lm_dst="$lm_dst/$fmt_lm"
fmt_beta="tmp/$(basename $fmt_am_nbest).beta"
beta=0.0
eval_sets="dev_clean dev_other test_clean test_other tlv2-dev tlv2-test"
keeplog=False

if [ 1 -eq "$(echo "${beta} != 0.0" | bc)" ]; then
    for subset in $eval_sets; do
        echo "$(printf $fmt_beta $subset).$beta"
        [ ! -f "$(printf $fmt_beta $subset).$beta" ] &&
            python utils/interpolate_nbests.py \
                "$(printf $fmt_beta $subset).$beta" \
                --nbestlist $(printf $fmt_beta $subset) \
                --weight $beta &
    done
    wait
    export fmt_beta="$fmt_beta.$beta"
    export search="0 1 1 0"
else
    export fmt_beta=""
    export search="0 1 1"
fi

for subset in $eval_sets; do
    log_dst="search-${subset}-$(echo $fmt_lm_src | cut -d '/' -f 2)--$(echo $fmt_lm_dst | cut -d '/' -f 2).log"
    python utils/lmweight_search.py \
        --nbestlist $(printf $fmt_am_nbest $subset) \
        $(printf $fmt_lm_src $subset) $(printf $fmt_lm_dst $subset) \
        $(printf "$fmt_beta" $subset) --search $search \
        --ground-truth ../../tools/CAT/egs/$(basename $(pwd))/data/$subset/text \
        --a-range -1 0 --a-int 0.02 --b-range 0 1 --b-int 0.02 \
        >$log_dst &
done
wait

for subset in $eval_sets; do
    log_dst="search-${subset}-$(echo $fmt_lm_src | cut -d '/' -f 2)--$(echo $fmt_lm_dst | cut -d '/' -f 2).log"
    grep SER $log_dst | grep -v tunning
    [ $keeplog == "False" ] && rm $log_dst
done
