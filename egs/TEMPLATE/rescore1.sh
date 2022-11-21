
score_path="exp/aishell-trf-fc-std/rescore/score-outd.nbest"
len_path="data/aishell/beta_old.nbest"
origin_path="data/aishell/outd.nbest"
gt_path="data/aishell/text"
for step_num in {1000..96000..1000}
do
    check_path=exp/aishell-trf-fc-std/check/checkpoint.*e${step_num}s.pt
    python change_infer_check.py exp/aishell-trf-fc-std $check_path
    python utils/pipeline/lm.py exp/aishell-trf-fc-std/ --sta 4
    python utils/lm/lmweight_search.py --nbestlist $origin_path $score_path $len_path --search 0 1 1 --ground $gt_path --cer
done


