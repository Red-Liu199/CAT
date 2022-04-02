"""
Prepare kaldi-like transcript and FBank features using torchaudio.
"""

import os
import sys
from typing import List, Dict, Callable, Any, Union, Tuple, Optional
from tqdm import tqdm
from copy import deepcopy

import kaldiio

import torch
import torchaudio


class Processor:
    """
    Processor to read the file and process the audio waveform.
    """

    def __init__(self, process_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._process_fn = process_fn
        self._next = []     # type: List[Processor]

    def __call__(self, inarg: Any) -> torch.Tensor:
        output = self._process_fn(inarg)
        for p_ in self._next:
            output = p_(output)
        return output

    def append(self, processor: "Processor"):
        self._next.append(processor)
        self._check_loop_ref()
        return self

    def clone(self) -> "Processor":
        """Clone self, would make a deep copy of process_fn but shallow copy of _next"""
        new_processor = Processor(deepcopy(self._process_fn))
        new_processor._next = self._next[:]
        return new_processor

    def _check_loop_ref(self):
        """Raise error if loop reference is found"""
        max_depth = 20
        depth = 0
        toexpand = [self]
        while toexpand != []:
            if depth >= max_depth:
                raise RuntimeError(
                    f"found reference depth over {max_depth}, possibly a loop reference.")

            if all(x._next == [] for x in toexpand):
                break
            else:
                depth += 1
                toexpand = sum([x._next for x in toexpand], [])


class ReadProcessor(Processor):
    """Processor wrapper to read from audio file."""

    def __init__(self) -> None:
        super().__init__(lambda file: torchaudio.load(file)[0])


def process_feat_as_kaldi(raw_audios: Dict[str, str], f_scp: str, processor: Processor):
    f_ark = f"{f_scp.removesuffix('.scp')}.ark"
    with kaldiio.WriteHelper(f'ark,scp:{f_ark},{f_scp}') as writer:
        for uid, _audio in tqdm(raw_audios.items()):
            writer(uid, processor(_audio).numpy())


def prepare_kaldi_feat(
        subsets: List[str],
        trans: Dict[str, List[Tuple[str, str]]],
        audios: Dict[str, List[str]],
        num_mel_bins: int = 80,
        sample_frequency: Optional[int] = None,
        speed_perturb: Optional[List[float]] = [],
        fmt_scp: str = "data/src/all_ark/{}.scp",
        fmt_trans: str = "data/src/{}/text"):

    subsets = list(set(subsets))
    for _set in subsets:
        assert _set in trans
        assert _set in audios

    if sample_frequency is None:
        sample_frequency = torchaudio.load(
            next(iter(audios[subsets[0]].values())))[1]

    fbank_processor = Processor(
        lambda waveform: torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=sample_frequency,
            num_mel_bins=num_mel_bins))
    audio2fbank = ReadProcessor().append(fbank_processor)

    for _set in subsets:
        f_trans = fmt_trans.format(_set)
        f_scp = fmt_scp.format(_set)
        os.makedirs(os.path.dirname(f_trans), exist_ok=True)
        os.makedirs(os.path.dirname(f_scp), exist_ok=True)

        # write transcript
        if os.path.isfile(f_trans):
            sys.stderr.write(
                f"warning: transcript {f_scp} exists, skip.\n")
        else:
            with open(f_trans, 'w') as fo:
                for uid, utt in trans[_set]:
                    fo.write(f"{uid}\t{utt}\n")

        # write feats
        if os.path.isfile(f_scp):
            sys.stderr.write(
                f"warning: scp file {f_scp} exists, skip.\n")
        else:
            process_feat_as_kaldi(audios[_set], f_scp, audio2fbank)

    for _factor in speed_perturb:
        if _factor == 1.0:
            continue
        sp_processor = Processor(
            lambda file: torchaudio.sox_effects.apply_effects_file(
                file, [['speed', f'{_factor:.5f}']])[0]).append(fbank_processor)
        for _set in subsets:
            f_trans = fmt_trans.format(f"{_set}-sp{_factor}")
            f_scp = fmt_scp.format(f"{_set}-sp{_factor}")
            os.makedirs(os.path.dirname(f_trans), exist_ok=True)
            os.makedirs(os.path.dirname(f_scp), exist_ok=True)
            # write trans
            if os.path.isfile(f_trans):
                sys.stderr.write(
                    f"warning: transcript {f_scp} exists, skip.\n")
            else:
                with open(f_trans, 'w') as fo:
                    for uid, utt in trans[_set]:
                        fo.write(f"{uid}#sp{_factor}\t{utt}\n")
            # write feats
            if os.path.isfile(f_scp):
                sys.stderr.write(
                    f"warning: scp file {f_scp} exists, skip.\n")
            else:
                process_feat_as_kaldi(audios[_set], f_scp, sp_processor)
