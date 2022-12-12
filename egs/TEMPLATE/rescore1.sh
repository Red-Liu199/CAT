score_path="exp/aishell-lm-fp-short/rescore/score-steaming_0.nbest"
len_path="data/aishell/beta.nbest"
origin_path="data/aishell/rnnt_bs16_best-10_streaming_False_test.nbest"
gt_path="data/aishell/text"
python utils/lm/lmweight_search.py --nbestlist $origin_path $score_path $len_path --search 0 1 1 --ground $gt_path --cer

score_path="exp/aishell-lm-fp-short/rescore/score-steaming_1.nbest"
len_path="data/aishell/beta.nbest"
origin_path="data/aishell/rnnt_bs16_best-10_streaming_True_test.nbest"
gt_path="data/aishell/text"
python utils/lm/lmweight_search.py --nbestlist $origin_path $score_path $len_path --search 0 1 1 --ground $gt_path --cer

