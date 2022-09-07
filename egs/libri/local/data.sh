#!/bin/bash                                                                                                                                                        
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# This script prepare libri data by torchaudio
set -e -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)
set -e -u
<<"PARSER"
("-src", type=str, default="/mnt/nas_workspace2/spmiData/librispeech",
    help="Source data folder containing the audios and transcripts. "
        "Download from https://www.openslr.org/12")
PARSER
eval $(python utils/parseopt.py $0 $*)

python local/extract_meta.py $src/LibriSpeech
python utils/data/resolvedata.py

echo "$0 done."
exit 0
