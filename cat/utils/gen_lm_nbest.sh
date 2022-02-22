set -u
set -e

usecpu=True
lm_dir=$1
checkpoint="avg_best_10.pt"
tokenizer="exp/lm-v13-v10-continue/tokenizer.tknz"
fmt_am_nbest="exp/rnnt-v27/decode/rnnt-128_algo-rna_nolm_best_10_%s.nbest"

eval_sets="dev_clean dev_other test_clean test_other tlv2-dev tlv2-test"

fmt_lm="$lm_dir/rescore/$(echo $fmt_am_nbest | cut -d '/' -f 2)"
mkdir -p $fmt_lm
fmt_lm="$fmt_lm/$(basename $fmt_am_nbest).lm"

[ ! -f $tokenizer ] && echo "No such tokenizer file found: $tokenizer" && exit 1

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
        cache_file="/tmp/$(basename $fmt_am_nbest).tmp"
        [ ! -f $(printf $fmt_lm $subset) ] &&
            python -m cat.lm.rescore \
                $(printf $fmt_am_nbest $subset) /dev/null \
                --config $lm_dir/config.json --resume $lm_dir/checks/$checkpoint \
                --tokenizer $tokenizer --save-lm-nbest $(printf $fmt_lm $subset)
    done
    wait
fi
