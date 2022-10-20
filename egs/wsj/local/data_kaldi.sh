#!/bin/bash
# author: Huahuan Zheng
# this script prepare the aishell data by kaldi tool
set -e

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("wsj0", type=str, nargs='?', default="/mnt/nas_workspace2/spmiData/WSJ/csr_1",
    help="Source data folder WSJ0.")
("wsj1", type=str, nargs='?', default="/mnt/nas_workspace2/spmiData/WSJ/csr_2_comp",
    help="Source data folder WSJ1.")
("-skip-meta-check", action="store_true",
    help="Skip the meta data preparation checking.")
("-use-3way-sp", action="store_true",
    help="Use 3-way speed perturbation.")
PARSER
eval $(python utils/parseopt.py $0 $*)

[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set. re-run with"
    echo "KALDI_ROOT=xxx $0 $*"
    exit 1
}
export KALDI_ROOT=$KALDI_ROOT

if [ $use_3way_sp == "True" ]; then
    opt_3way_sp="--apply-3way-speed-perturbation"
else
    opt_3way_sp=""
fi

if [ $skip_meta_check == "False" ]; then
    # extract meta
    echo "The meta data preparation of the WSJ dataset is rather complex,"
    echo "... please refer to $KALDI_ROOT/egs/wsj/s5 for kaldi processing."
    echo "Once it is done. Link the dir 'wsj/s5/data' to 'data/src' via:"
    echo "    ln -s \$(readlink -f wsj/s5/data) data/src"
    echo "Then re-run this script with $0 $* -skip-meta-check"
    exit 1
fi

# prepare extra corpus
[ ! -f data/extra.corpus ] &&
    zcat $wsj1/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z |
    grep -v "<" |
        tr "[:lower:]" "[:upper:]" |
        >data/extra.corpus

# compute fbank feat
# by default, we use 80-dim raw fbank and do not apply
# ... the CMVN, which matches the local/data.sh via torchaudio
bash utils/data/data_prep_kaldi.sh \
    data/src/{train_si284,test_dev93,test_eval92} \
    --feat-dir=data/fbank \
    --nj=$(nproc) \
    --not-apply-cmvn \
    $opt_3way_sp

# prepare cmu dict in case of usage
[ ! -f data/cmudict.txt ] &&
    wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/sphinxdict/cmudict.0.7a_SPHINX_40 \
        -O data/cmudict.txt

# refresh the data/metainfo.json file
python utils/data/resolvedata.py

echo "$0 done."
echo "go and check data/metainfo.json for dataset info."
echo "CMU dict: data/cmudict.txt"
exit 0
