#!/bin/bash
# author: Huahuan Zheng
# this script prepare the aishell data by kaldi tool
set -e

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="/mnt/nas_workspace2/spmiData/aishell_data",
    help="Source data folder containing the audios and transcripts. "
        "Download from https://www.openslr.org/resources/33/data_aishell.tgz")
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

# extract meta
bash local/extract_meta_kaldi.sh $src/wav \
    $src/transcript/aishell_transcript_v0.8.txt

# compute fbank feat
# by default, we use 80-dim raw fbank and do not apply
# ... the CMVN, which matches the local/data.sh via torchaudio
bash utils/data/data_prep_kaldi.sh \
    data/src/{train,dev,test} \
    --feat-dir=data/fbank \
    --nj=16 \
    --not-apply-cmvn \
    $opt_3way_sp

# refresh the data/metainfo.json file
python utils/data/resolvedata.py

# remove spaces
for file in $(python -c "import json;\
print(' '.join(x['trans'] for x in json.load(open('data/metainfo.json', 'r')).values()))"); do
    [ ! -f $file.bak ] && mv $file $file.bak
    python utils/data/clean_space.py -i $file.bak -o $file || exit 1
done

echo "$0 done."
exit 0
