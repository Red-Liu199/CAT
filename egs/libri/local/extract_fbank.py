"""
Compute FBank feature for aishell using torchaudio.
"""


import os
import sys
import glob
import argparse
from typing import List, Dict

prepare_sets = [
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
    'train-clean-100',
    'train-clean-360',
    'train-other-500'
]

expect_len = {
    'dev-clean': 2703,
    'dev-other': 2864,
    'test-clean': 2620,
    'test-other': 2939,
    'train-clean-100': 28539,
    'train-clean-360': 104014,
    'train-other-500': 148688
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="/data/librispeech/LibriSpeech",
                        help="Directory to source audio files, "
                        f"expect sub-dir: {', '.join(prepare_sets)} in the directory.")
    args = parser.parse_args()

    assert os.path.isdir(args.src_data)
    for _set in prepare_sets:
        assert os.path.isdir(os.path.join(args.src_data, _set)
                             ), f"subset '{_set}' not found in {args.src_data}"

    os.makedirs('data/src')
    trans = {}      # type: Dict[str, List[str]]
    audios = {}     # type: Dict[str, Dict[str, str]]
    for _set in prepare_sets:
        d_audio = os.path.join(args.src_data, _set)
        _audios = glob.glob(f"{d_audio}/**/**/*.flac")
        trans[_set] = []
        for f_ in sorted(glob.glob(f"{d_audio}/**/**/*.trans.txt")):
            with open(f_, 'r') as fi:
                for line in fi:
                    uid, utt = line.strip().split(maxsplit=1)
                    trans[_set].append((uid, utt))

        audios[_set] = {}
        for _raw_wav in _audios:
            uid = os.path.basename(_raw_wav).removesuffix('.flac')
            audios[_set][uid] = _raw_wav

        assert len(audios[_set]) == len(trans[_set]), \
            f"# audio mismatches # transcript in {_set}: {len(audios[_set])} != {len(trans[_set])}"
        if len(audios[_set]) != expect_len[_set]:
            sys.stderr.write(
                f"warning: found {len(audios[_set])} audios in {_set} subset, but expected {expect_len[_set]}")

    try:
        import cat
    except ModuleNotFoundError:
        import sys
        import os
        sys.path.append(os.getcwd())
    from cat.utils.data_prep_kaldi import prepare_kaldi_feat
    prepare_kaldi_feat(
        subsets=prepare_sets,
        trans=trans,
        audios=audios,
        num_mel_bins=80,
        speed_perturb=[]
    )
