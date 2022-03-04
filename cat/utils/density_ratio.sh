set -u
set -e

lm_src=$1
lm_dst=$2

fmt_am_nbest="exp/rnnt-v27/decode/rnnt-128_algo-rna_nolm_best_10_%s.nbest"

fmt_lm="rescore/$(echo $fmt_am_nbest | cut -d '/' -f 2)/$(basename $fmt_am_nbest).lm"
fmt_lm_src="$lm_src/$fmt_lm"
fmt_lm_dst="$lm_dst/$fmt_lm"
fmt_beta=$(echo $fmt_lm_dst | sed 's/[.]lm/.beta/g')
beta=0.0
# eval_sets="dev_clean dev_other test_clean test_other tlv2-dev tlv2-test"
eval_sets="test_clean test_other tlv2-dev tlv2-test"

for subset in $eval_sets; do
    python utils/lmweight_search.py \
        --nbestlist $(printf $fmt_am_nbest $subset) \
        $(printf $fmt_lm_src $subset) $(printf $fmt_lm_dst $subset) \
        $(printf "$fmt_beta" $subset) \
        --search 0 1 1 0 --weight 1.0 $beta \
        --ground-truth ../../tools/CAT/egs/$(basename $(pwd))/data/$subset/text \
        --a-range -1 0 --a-int 0.02 --b-range 0 1 --b-int 0.02
done | grep SER | grep -v tunning
wait
