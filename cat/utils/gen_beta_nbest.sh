set -u
set -e

eval_sets="dev_clean dev_other test_clean test_other tlv2-dev tlv2-test"

fmt_am_nbest="exp/rnnt-v27/decode/rnnt-128_algo-rna_nolm_best_10_%s.nbest"
tokenizer="exp/lm-v13-v10-continue/tokenizer.tknz"

mkdir -p tmp/
fmt_beta="tmp/$(basename $fmt_am_nbest).beta"

[ ! -f $tokenizer ] && echo "No such tokenizer file found: $tokenizer" && exit 1

for subset in $eval_sets; do
    echo $(printf $fmt_beta $subset)
    [ ! -f $(printf $fmt_beta $subset) ] &&
        python utils/interpolate_nbests.py $(printf $fmt_beta $subset) \
            --nbestlist $(printf $fmt_am_nbest $subset) --weight 0.0 \
            --ins-penalty 1.0 --tokenizer $tokenizer &
done
wait
