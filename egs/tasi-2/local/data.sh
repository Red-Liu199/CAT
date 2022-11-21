#!/bin/bash
# author: Huahuan Zheng

set -e
set -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="/mnt/nas3_workspace/spmiData/tasi2_16k",
    help="Source data folder containing the audios and transcripts. ")
("-out", type=str, default="/data",
    help="Output directory.")
("-subsets-fbank", type=str, nargs="+", default=[],
    help="Subset(s) for extracting FBanks. Default: all")
PARSER
eval $(python utils/parseopt.py $0 $*)

[ ! -d $src/train ] && {
    echo "No such audio files dir: $src/train"
    exit 1
}

# text normalize
trans=$src/trans.noseg
[ ! -f $trans ] && {
    echo "Normalized transcript file no found: '$trans'"
    echo "... would generate from source."
    export trans="./trans.noseg"
    bash local/text_normalize.sh \
        $src/tasi2_8k.txt \
        -out $trans || exit 1
}

# Extract 80-dim FBank features
# here only the wav-clean is used, you can
# prepare the wav-noise in the same way.
python local/extract_meta.py \
    $src/train/wav_clean \
    $trans $out \
    --subset $subsets_fbank || exit 1

echo "$0 done"
exit 0
