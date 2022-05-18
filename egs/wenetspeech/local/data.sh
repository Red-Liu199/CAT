# Prepare the WenetSpeech data in kaldi format.
# If you want to prepare in WebDataset format, please refer to local/prep_wds.py

set -e
set -u

data="/mnt/nas3_workspace/spmiData/WenetSpeech"

[ ! -d $data ] && (
    echo "Download the data from https://wenet.org.cn/WenetSpeech/ "
    exit 1
)

[ ! $(command -v python) ] && (
    echo "no python interpreter in your PATH"
    exit 1
)

[ ! -f $data/WenetSpeech.json ] && (
    echo "meta file: $data/WenetSpeech.json not found. Link it to there if you put it elsewhere."
    exit 1
)
python local/extract_meta.py \
    $data $data/WenetSpeech.json \
    -o data/thaudio || exit 1

python local/prep_fbank.py data/thaudio \
    --subset M DEV TEST_MEETING TEST_NET \
    --cmvn || exit 1
