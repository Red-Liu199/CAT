exp_no=$1
python utils/rescore_std.py \
  --dev_asr "data/wenet/dev.nbest" \
  --dev_lm "${exp_no}/rescore/score_dev.nbest"\
  --dev_beta "data/wenet/beta_dev.nbest"\
  --dev_text "data/wenet/text_dev"\
  --test_asr "data/wenet/test_meeting.nbest" "data/wenet/test_net.nbest"\
  --test_lm "${exp_no}/rescore/score_meeting.nbest" "${exp_no}/rescore/score_net.nbest"\
  --test_beta "data/wenet/beta_meeting.nbest" "data/wenet/beta_net.nbest"\
  --test_text "data/wenet/text_meeting" "data/wenet/text_net"
