set -e
set -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="/mnt/nas3_workspace/spmiData/tasi2_16k",
    help="Source data folder containing the audios and transcripts. ")
("-out", type=str, default="/mnt/nas3_workspace/zhenghh/speechdata/tasi2_16k",
    help="Output directory.")
("-subsets-fbank", type=str, nargs="+", default=[],
    help="Subset(s) for extracting FBanks. Default: all")
PARSER
eval $(python utils/parseopt.py $0 $*)

# Extract 80-dim FBank features
python local/extract_meta.py \
    $src/wav/wav_clean $src/trans.noseg $out \
    --subset $subsets_fbank || exit 1

echo "$0 done"
exit 0
