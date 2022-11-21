# Prepare the WenetSpeech data in kaldi format.
# If you want to prepare in WebDataset format, please refer to local/prep_wds.py

set -e -u

[ ! -z $* ] && {
    echo "$0 accept no arguments."
    exit 1
}

data="/mnt/nas3_workspace/spmiData/WenetSpeech"
nj=48

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

[ ! -f data/meta/.done ] && {
    python local/extract_meta.py \
        $data $data/WenetSpeech.json \
        -o data/meta || exit 1
    touch data/meta/.done
}
echo "Meta data extracted."

python local/prep_fbank.py data/meta \
    --subset M DEV TEST_MEETING TEST_NET \
    --nj=$nj || exit 1
echo "Feature computed."

python utils/data/resolvedata.py

echo "$0 done."
exit 0
