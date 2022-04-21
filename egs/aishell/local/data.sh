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
    $srcdata/transcript/aishell_transcript_v0.8.txt || exit 1

# do speed perturbation
python local/extract_fbank.py $srcdata/wav \
    $srcdata/transcript/aishell_transcript_v0.8.txt \
    --subset train --speed-perturbation 0.9 1.1 || exit 1

python utils/data/resolvedata.py

# remove spaces
for file in $(python -c "import json;\
print(' '.join(x['trans'] for x in json.load(open('data/metainfo.json', 'r')).values()))"); do
    [ ! -f $file.bak ] && mv $file $file.bak
    python utils/data/clean_space.py -i $file.bak -o $file || exit 1
done

echo "$0 done"
exit 0
