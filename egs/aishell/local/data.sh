set -e
set -u

srcdata="/mnt/nas_workspace2/spmiData/aishell_data/"
# or download the data from
# https://www.openslr.org/resources/33/data_aishell.tgz

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

python local/extract_fbank.py $srcdata/wav \
    $srcdata/transcript/aishell_transcript_v0.8.txt \
    --speed-perturbation 0.9 1.1
