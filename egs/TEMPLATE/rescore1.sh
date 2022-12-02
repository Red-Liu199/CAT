exp_no=$1
score_path="exp/${exp_no}/rescore/score-ind.nbest"
len_path="data/aishell/beta.nbest"
origin_path="data/aishell/ind.nbest"
gt_path="data/aishell/text"
python utils/lm/lmweight_search.py --nbestlist $origin_path $score_path $len_path --search 0 1 1 --ground $gt_path --cer

