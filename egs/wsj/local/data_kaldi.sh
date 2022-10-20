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
    --nj=48 \
    --not-apply-cmvn \
    $opt_3way_sp

# do some text normalize
for dset in $(ls data/src); do
    dir="data/src/$dset"
    [[ $dset != "test_*" ]] && [[ -f $dir/text ]] && {
        cut <$dir/text -d ' ' -f 1 >$dir/text.filt.id.tmp
        cut <$dir/text -d ' ' -f 2- | sed \
            -e 's/[(]BRACE//g; s/[(]PAREN//g; s/[(]IN-PARENTHESIS//g; s/[(]BEGIN-PARENS//g' \
            -e 's/[(]PARENTHESES//g; s/[(]LEFT-PAREN//g; s/[(]PARENTHETICALLY//g' \
            -e 's/[)]PAREN//g; s/[)]RIGHT-PAREN//g; s/[)]END-OF-PAREN//g; s/[)]END-PARENS//g;' \
            -e 's/[)]END-THE-PAREN//g; s/[)]CLOSE-BRACE//g; s/[)]CLOSE_PAREN//g; s/[)]CLOSE-PAREN//g; s/[)]UN-PARENTHESES//g' \
            -e 's/"CLOSE-QUOTE//g; s/"DOUBLE-QUOTE//g; s/"END-OF-QUOTE//g; s/"END-QUOTE//g; s/"IN-QUOTES//g; s/"QUOTE//g; s/"UNQUOTE//g' \
            -e 's/{LEFT-BRACE//g; s/}RIGHT-BRACE//g; s/\~//g' \
            -e 's/;SEMI-COLON//g; s/\/SLASH//g; s/&AMPERSAND/AND/g' \
            -e 's/<\*IN\*>//g; s/<\*MR\.\*>//g; s/<NOISE>//g' \
            -e 's/[.]PERIOD//g; s/[.]DOT//g; s/\.\.\.ELLIPSIS//g;' \
            -e 's/[*]//g; s/:COLON/:/g; s/?QUESTION-MARK//g' \
            -e "s/'SINGLE-QUOTE//g; " \
            -e 's/[(]//g; s/[)]//g; s/-//g; s/;//g; s/[.]//g; s/!//g; ' \
            -e 's/://g; s/`//g; s/[.]//g; s/,COMMA//g; s/?//g' \
            >$dir/text.filt.ct.tmp
        paste $dir/text.filt.{id,ct}.tmp >$dir/text.filt
        rm -f $dir/text.filt.{id,ct}.tmp
        [ ! -f $dir/text.orin ] &&
            mv $dir/text{,.orin}
        mv $dir/text{.filt,}
    }
done

# refresh the data/metainfo.json file
python utils/data/resolvedata.py

echo "$0 done."
exit 0
