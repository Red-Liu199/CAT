"""
Compute FBank feature for aishell using torchaudio.
"""

import os
import sys
import glob
import math
import argparse
from typing import List, Dict
from tqdm import tqdm

import kaldiio

import torch
import torchaudio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="/data/AISHELL-1/wav",
                        help="Directory to source audio files, expect sub-dir 'train', 'dev' and 'test' in the directory.")
    parser.add_argument("transcript", type=str, default="/data/transcript/aishell_transcript_v0.8.txt",
                        help="Path to the transcript file.")
    args = parser.parse_args()

    assert os.path.isfile(
        args.transcript), f"Trancript: '{args.transcript}' does not exist."
    assert os.path.isdir(args.src_data)
    for _set in ['train', 'dev', 'test']:
        assert os.path.isdir(os.path.join(args.src_data, _set))

    os.makedirs('data/src', exist_ok=True)
    trans = {}      # type: Dict[str, str]
    with open(args.transcript, 'r') as fi:
        for line in fi:
            uid, utt = line.strip().split(maxsplit=1)
            trans[uid] = utt

    expect_len = {
        'train': 120098,
        'dev': 14326,
        'test': 7176
    }
    audios = {}     # type: Dict[str, Dict[str, str]]
    subtrans = {}   # type: Dict[str, List[str]]
    for _set in ['train', 'dev', 'test']:
        d_audio = os.path.join(args.src_data, _set)
        _audios = glob.glob(f"{d_audio}/**/*.wav")
        audios[_set] = {}
        subtrans[_set] = []
        for _raw_wav in _audios:
            uid = os.path.basename(_raw_wav).removesuffix('.wav')
            if uid not in trans:
                continue
            audios[_set][uid] = _raw_wav
            subtrans[_set].append(f"{uid}\t{trans[uid]}")
        if len(audios[_set]) != expect_len[_set]:
            sys.stderr.write(
                f"warning: found {len(audios[_set])} audios in {_set} subset, but expected {expect_len[_set]}")
    del trans

    _, sample_frequency = torchaudio.load(next(iter(audios['train'].values())))
    num_mel_bins = 80
    scp_dir = "data/src/all_ark"
    os.makedirs(scp_dir, exist_ok=True)
    for _set in audios.keys():
        # write transcript
        os.makedirs(f"data/src/{_set}", exist_ok=True)
        with open(f"data/src/{_set}/text", 'w') as fo:
            fo.write('\n'.join(sorted(subtrans[_set])))

        # write feats
        with kaldiio.WriteHelper(f'ark,scp:{scp_dir}/{_set}.ark,{scp_dir}/{_set}.scp') as writer:
            for uid, _audio in tqdm(audios[_set].items()):
                writer(uid, torchaudio.compliance.kaldi.fbank(
                    torchaudio.load(_audio)[0],
                    sample_frequency=sample_frequency,
                    num_mel_bins=num_mel_bins).numpy())
