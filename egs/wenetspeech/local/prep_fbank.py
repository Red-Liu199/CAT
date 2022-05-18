"""
Prepare the FBank feats.
Author: Zheng Huahuan
"""


import os
import pickle
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

import torch
import kaldiio


valid_subsets = [
    'DEV',
    'L',
    'M',
    'S',
    'W',
    'TEST_MEETING',
    'TEST_NET'
]


def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help="Directory that contains the extraced meta info.")
    parser.add_argument("--subset", type=str, nargs='+', choices=valid_subsets, default=['M', 'DEV', 'TEST_MEETING', 'TEST_NET'],
                        help="Subsets to be prepared.")
    parser.add_argument("--cmvn", action="store_true",
                        help="Apply CMVN by utterance.")
    args = parser.parse_args()

    annotations = {}
    args.subset = list(set(args.subset))
    for _s in args.subset:
        assert _s in valid_subsets
        if _s not in ['DEV', 'TEST_NET', 'TEST_MEETING']:
            annotations[_s] = 'train_'+_s.lower()
        else:
            annotations[_s] = _s.lower()

    with open(f"{args.data}/subsetlist.pkl", 'rb') as fib:
        subset2utt = pickle.load(fib)      # type: Dict[str, List[str]]

    for _s in list(subset2utt.keys()):
        if _s not in args.subset:
            del subset2utt[_s]

    with open(f"{args.data}/audiodur.pkl", 'rb') as fib:
        # type: Dict[str, List[Tuple[str, int, int]]]
        audio2dur = pickle.load(fib)
        utt2audio = pickle.load(fib)    # type: Dict[str, str]
    # filter the audio2dur
    uttlist = sum(subset2utt.values(), [])
    audiofiles = {
        utt2audio[_utt]: None
        for _utt in uttlist
    }
    audio2dur = {aid: audio2dur[aid] for aid in audiofiles}

    with open(os.path.join(args.data, 'wav'), 'r') as fit_wavs:
        for line in fit_wavs:
            aid, path = line.strip().split()
            if aid in audiofiles:
                audiofiles[aid] = path

    ark_file = 'data/.ark/feats.ark'
    scp_file = 'data/.ark/feats.scp'
    if not os.path.exists('data/.ark/.done'):
        from cat.utils.data.data_prep_kaldi import *

        os.makedirs(os.path.dirname(ark_file), exist_ok=True)
        assert not os.path.isfile(ark_file), ark_file
        processor = ReadProcessor().append(FBankProcessor(16000, 80))
        if args.cmvn:
            cmvn_processor = CMVNProcessor()
        with kaldiio.WriteHelper(f'ark,scp:{ark_file},{scp_file}') as writer:
            for aid, _audio in tqdm(audiofiles.items(), desc='Extract FBank', leave=False):
                feat = processor(_audio)    # type: torch.Tensor
                for uid, s_beg, s_end in audio2dur[aid]:
                    u_feat = feat[s_beg:s_end]
                    if args.cmvn:
                        u_feat = cmvn_processor(u_feat)
                    writer(uid, u_feat.numpy())

        touch('data/.ark/.done')
        print("> FBank extracted.")

    print("> Prepare text...")
    assert not os.path.exists('data/src')
    with open(f"{args.data}/corpus.pkl", 'rb') as fib:
        text_corpus = pickle.load(fib)      # type: Dict[str, str]

    for _set in args.subset:
        f_text = f'data/src/{annotations[_set]}/text'
        if os.path.exists(f_text):
            print(f"{f_text} exists, skip.")
            continue

        with open(f_text, 'w') as fot:
            for uid in subset2utt[_set]:
                fot.write(f"{uid}\t{text_corpus[uid]}")

    print("> done.")

    print("> Prepare scp file...")
    scps = {}
    with open(scp_file, 'r') as fit:
        for line in fit:
            uid, ark_loc = line.split(maxsplit=1)
            scps[uid] = ark_loc

    for _set in args.subset:
        f_scp = f"data/all_ark/{annotations[_set]}.scp"
        with open(f_scp, 'w') as fot:
            for uid in subset2utt[_set]:
                fot.write(f"{uid}\t{scps[uid]}")
    print("> done.")
