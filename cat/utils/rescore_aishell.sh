exp_no=$1
mode=$2
python rescore_std.py \
  --dev_asr "data/aishell/${mode}-dev.nbest" \
  --dev_lm "${exp_no}/rescore/score-${mode}-dev.nbest"\
  --dev_beta "data/aishell/${mode}-dev-beta.nbest"\
  --dev_text "data/aishell/dev-text"\
  --test_asr "data/aishell/${mode}.nbest"\
  --test_lm "${exp_no}/rescore/score-${mode}.nbest"\
  --test_beta "data/aishell/beta_${mode}.nbest"\
  --test_text "data/aishell/text"
