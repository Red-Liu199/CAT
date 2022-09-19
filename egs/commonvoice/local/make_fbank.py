"""
Compute FBank feature for commonvoice format source data. using torchaudio.
"""


import os
import sys
import argparse
from typing import List, Dict, Any, Tuple


prepare_sets = [
    'train',
    'validated',
    'dev',
    'test'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="data/src", nargs='?',
                        help="Directory includes the meta infos.")
    parser.add_argument("--subset", type=str, nargs='*',
                        choices=prepare_sets, help=f"Specify datasets in {prepare_sets}")
    parser.add_argument("--cmvn", action="store_true", default=False,
                        help="Apply CMVN by utterance, default: False.")
    parser.add_argument("--speed-perturbation", type=float, dest='sp',
                        nargs='*', default=[], help=f"Add speed perturbation to subset: {', '.join(prepare_sets)}")
    args = parser.parse_args()

    assert os.path.isdir(args.src_data)
    if args.subset is not None:
        for _set in args.subset:
            assert _set in prepare_sets, f"--subset {_set} not in predefined datasets: {prepare_sets}"
        prepare_sets = args.subset

    for _sp_factor in args.sp:
        assert (isinstance(_sp_factor, float) or isinstance(_sp_factor, int)) and _sp_factor > 0, \
            f"Unsupport speed pertubation value: {_sp_factor}"

    d_sets = [os.path.join(args.src_data, _set) for _set in prepare_sets]

    from cat.utils.data import data_prep
    data_prep.prepare_kaldi_feat(
        subsets=prepare_sets,
        trans=[f"{path}/text" for path in d_sets],
        audios=[f"{path}/wav.scp" for path in d_sets],
        num_mel_bins=80,
        apply_cmvn=args.cmvn,
        speed_perturb=args.sp,
        read_from_extracted_meta=True
    )
