set -e
set -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="/mnt/nas_workspace2/spmiData/aishell_data",
    help="Source data folder containing the audios and transcripts. "
        "Download from https://www.openslr.org/resources/33/data_aishell.tgz")
("-sp", type=float, nargs='*', default=None,
    help="Speed perturbation factor(s). Default: None.")
("-subsets-fbank", type=str, nargs="+", choices=["train", "dev", "test"],
    default=["train", "dev", "test"], help="Subset(s) for extracting FBanks. Default: ['train', 'dev', 'test']")
PARSER
eval $(python utils/parseopt.py $0 $*)

opt_sp="1.0"
[ "$sp" != "None" ] && export opt_sp=$sp

# Extract 80-dim FBank features
python local/extract_meta.py $src/wav \
    $src/transcript/aishell_transcript_v0.8.txt \
    --subset $subsets_fbank --speed-perturbation $opt_sp || exit 1

python utils/data/resolvedata.py

# remove spaces
for file in $(python -c "import json;\
print(' '.join(x['trans'] for x in json.load(open('data/metainfo.json', 'r')).values()))"); do
    [ ! -f $file.bak ] && mv $file $file.bak
    python utils/data/clean_space.py -i $file.bak -o $file || exit 1
done

echo "$0 done"
exit 0
