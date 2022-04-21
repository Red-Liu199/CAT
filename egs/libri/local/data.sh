set -e
set -u

srcdata="/mnt/nas_workspace2/spmiData/librispeech"
# or download the data from
# https://www.openslr.org/12/

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

python local/extract_fbank.py $srcdata/LibriSpeech
python utils/data/resolvedata.py

echo "$0 done."
exit 0
