"""
Prepare the FBank feats.
Author: Zheng Huahuan
"""

# FIXME: loading the .opus file with torchaudio is too slow, see https://github.com/pytorch/audio/issues/1994
try:
    import lhotse
except ModuleNotFoundError:
    print("Module 'lhotse' is not found. Install with:\n"
          "pip install lhotse")
finally:
    from lhotse.audio import read_opus

import os
import pickle
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

import torch
import kaldiio

from torch.utils.data import DataLoader, Dataset
from cat.utils.data.data_prep_kaldi import *


valid_subsets = [
    'DEV',
    'L',
    'M',
    'S',
    'W',
    'TEST_MEETING',
    'TEST_NET'
]


class OpusReadProcessor(Processor):
    def _process_fn(self, opus_file, *args, **kwargs) -> torch.Tensor:
        return torch.from_numpy(read_opus(opus_file, *args, **kwargs)[0])


class OpusData(Dataset):
    def __init__(self, f_utt2dur: str, uttlist: List[str], apply_cmvn=False) -> None:
        super().__init__()
        with open(f_utt2dur, 'rb') as fib:
            # type: Dict[str, Tuple[str, int, int]]
            utt2audio = pickle.load(fib)

        self._uttlist = uttlist
        self._uttinfo = [utt2audio[uid] for uid in uttlist]
        assert len(self._uttinfo) == len(self._uttlist)
        del utt2audio

        self.processor = (
            OpusReadProcessor()
            .append(FBankProcessor(16000, 80))
        )
        if apply_cmvn:
            self.cmvn_processor = CMVNProcessor()
        else:
            self.cmvn_processor = None

    def __len__(self) -> int:
        return len(self._uttlist)

    def __getitem__(self, index: int):
        uid = self._uttlist[index]
        path, s_beg, s_end = self._uttinfo[index]
        u_feat = self.processor(
            path,
            offset=s_beg,
            duration=(s_end-s_beg),
            force_opus_sampling_rate=16000
        )
        if self.cmvn_processor is not None:
            u_feat = self.cmvn_processor(u_feat)

        return uid, u_feat


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
    parser.add_argument("--nj", type=int, default=16,
                        help="Number of jobs to read the audios.")
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

    ark_file = 'data/.ark/feats.ark'
    scp_file = 'data/.ark/feats.scp'
    if not os.path.exists('data/.ark/.done'):
        assert not os.path.isfile(ark_file), ark_file

        uttlist = sum(subset2utt.values(), [])

        os.makedirs(os.path.dirname(ark_file), exist_ok=True)
        dataloader = DataLoader(OpusData(
            f_utt2dur=f"{args.data}/utt2dur.pkl",
            uttlist=uttlist,
            apply_cmvn=args.cmvn
        ), shuffle=False, num_workers=args.nj, batch_size=None)
        p_bar = tqdm(desc='Extract FBank', leave=False, total=len(uttlist))
        with kaldiio.WriteHelper(f'ark,scp:{ark_file},{scp_file}') as writer:
            for uid, feat in dataloader:
                writer(uid, feat.numpy())
                p_bar.update()
        p_bar.close()

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
