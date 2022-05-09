"""
Compute FBank feature for aishell using torchaudio.
"""

import webdataset as wds
from typing import List, Tuple, Dict
from cat.utils.data.data_prep_kaldi import *
import os
import sys
import glob
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


prepare_sets = [
    'accented_hearingdata',
    'aidatatang',
    'aishell',
    'aishell2',
    'Android',
    'data_863',
    'Datatang-Codeswitch',
    'Datatang-Conversation',
    'Datatang-Dialect',
    'Intel',
    'iOS',
    'magic_data',
    'Phrase',
    'st_cmds',
    'tal_aisolution',
    'tal_chinese',
    'tal_english',
    'tal_math',
    'WindowsMobile'
]


def pack_data(
        f_audios: List[str],     # audio file list
        label_dict: Dict[str, str],
        d_out: str,      # output directory
        fmt: str = "data-%05d.tar",       # tar file name format
        filter_group: List[str] = None):

    os.makedirs(d_out, exist_ok=True)

    # initialize filter bounds
    if filter_group is None:
        filter_group = [':']

    filter_bound = []   # type: List[Tuple[int, int]]
    for filt in filter_group:
        assert ':' in filt, f"pack_data: invalid filter format {filt}"
        l_bound, u_bound = filt.split(':')
        l_bound = 1 if l_bound == '' else int(l_bound)
        u_bound = 2**32-1 if u_bound == '' else int(u_bound)
        filter_bound.append((l_bound, u_bound))
    # it's your duty to ensure the intervals are not overlap
    filter_bound = sorted(filter_bound, key=lambda item: item[0])
    del filter_group

    # initialize wds writer groups
    if len(filter_bound) > 1:
        sinks = []
        for l, u in filter_bound:
            prefix = '' if l <= 1 else str(l)
            suffix = '' if u == (2**32-1) else str(u)
            subdir = os.path.join(d_out, f"{prefix}_{suffix}")
            os.makedirs(subdir, exist_ok=True)
            sinks.append(
                wds.ShardWriter(os.path.join(subdir, fmt), maxcount=2000)
            )
    else:
        sinks = [wds.ShardWriter(os.path.join(d_out, fmt), maxcount=2000)]

    # initialize processor
    processor = ReadProcessor().append(FBankProcessor(16000, 80))
    for _audio in tqdm(f_audios):
        fbank_spec = processor(_audio)

        uid = os.path.basename(_audio).removesuffix('.wav')
        target = label_dict.get(uid, None)
        if target is None:
            sys.stderr.write(
                f"Fount audio file: '{_audio}', but no match transcription for uid: '{uid}'. skip\n"
            )
            continue

        L = fbank_spec.shape[0]

        for (l, u), _sink in zip(filter_bound, sinks):
            if l <= L and L < u:
                try:
                    _sink.write({
                        '__key__': uid,
                        "mat.npy": fbank_spec.numpy(),
                        "label.txt": target
                    })
                except Exception as e:
                    if "Error loading audio" in str(e):
                        sys.stderr.write(
                            f"error loading audio: {str(e)}, skip\n")
                finally:
                    break

    for s in sinks:
        s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="/mnt/nas3_workspace/spmiData/tasi2_16k/wav/wav_clean",
                        help="Directory to source audio files, "
                        f"expect subset: {', '.join(prepare_sets)} in the directory.")
    parser.add_argument("transcript", type=str, default="/mnt/nas3_workspace/spmiData/tasi2_16k/tasi2_8k.txt",
                        help="Path to the transcript file.")
    parser.add_argument("odir", type=str, help="Ouput directory.")
    parser.add_argument("--subset", type=str, nargs='*', default=[],
                        choices=prepare_sets, help=f"Specify datasets in {prepare_sets}")
    args = parser.parse_args()

    assert os.path.isfile(
        args.transcript), f"Trancript: '{args.transcript}' does not exist."
    assert os.path.isdir(args.src_data)
    if args.subset != []:
        for _set in args.subset:
            assert _set in prepare_sets, f"--subset {_set} not in predefined datasets: {prepare_sets}"
        prepare_sets = args.subset

    for _set in prepare_sets:
        assert os.path.isdir(os.path.join(args.src_data, _set))

    assert os.access(
        args.odir, os.W_OK), f"permission denied to write '{args.odir}'"

    trans = {}      # type: Dict[str, str]
    with open(args.transcript, 'r') as fi:
        for line in fi:
            contents = line.strip().split(maxsplit=1)
            if len(contents) < 2:
                continue
            uid, utt = contents
            trans[uid] = utt

    for _set in prepare_sets:
        pack_data(
            f_audios=glob.glob(os.path.join(args.src_data, f"{_set}/*.wav")),
            label_dict=trans,
            d_out=os.path.join(args.odir, _set),
            fmt=(_set+r"-%05d.tar"),
            filter_group=[":10", "10:1000", "1000:1200",
                          "1200:1500", "1500:2000", "2000:"]
        )
