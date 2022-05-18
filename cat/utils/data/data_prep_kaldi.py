"""
Prepare kaldi-like transcript and FBank features using torchaudio.
"""

import os
import sys
from typing import *
from tqdm import tqdm

import kaldiio

import torch
import torchaudio
import torchaudio.transforms as T

__all__ = ["Processor", "ReadProcessor", "ResampleProcessor",
           "SpeedPerturbationProcessor", "FBankProcessor", "CMVNProcessor"]


class Processor:
    """
    Processor to read the file and process the audio waveform.
    """

    def __init__(self) -> None:
        self._next = []     # type: List[Processor]

    def _process_fn(self, inarg: Any) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = self._process_fn(*args,  **kwargs)
        for p_ in self._next:
            output = p_(output)
        return output

    def append(self, processor: "Processor"):
        self._next.append(processor)
        self._check_loop_ref()
        return self

    def clone(self) -> "Processor":
        new_processor = Processor()
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

    def _process_fn(self, file: str, *args, **kwargs) -> torch.Tensor:
        return torchaudio.load(file, *args, **kwargs)[0]


class SpeedPerturbationProcessor(Processor):
    """Processor wrapper to do speed perturbation"""

    def __init__(self, factor: float) -> None:
        super().__init__()
        assert isinstance(factor, float) or isinstance(factor, int)
        assert factor > 0

        self._sp_factor = factor

    def _process_fn(self, file: str) -> torch.Tensor:
        return torchaudio.sox_effects.apply_effects_file(
            file, [['speed', f'{self._sp_factor:.5f}']])[0]


class FBankProcessor(Processor):
    """Processor wrapper to compute FBank feat"""

    def __init__(self, sample_rate: int, num_mel_bins: int) -> None:
        super().__init__()
        assert isinstance(sample_rate, int)
        assert sample_rate > 0
        assert isinstance(num_mel_bins, int)
        assert num_mel_bins > 0

        self._sample_rate = sample_rate
        self._num_mel_bins = num_mel_bins

    def _process_fn(self, waveform: torch.Tensor) -> torch.Tensor:
        return torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=self._sample_rate,
            num_mel_bins=self._num_mel_bins)


class ResampleProcessor(Processor):
    def __init__(self, orin_sample_rate: int, new_sample_rate: int) -> None:
        super().__init__()
        self._orin_rate = orin_sample_rate
        self._new_rate = new_sample_rate
        self.resampler = T.Resample(orin_sample_rate, new_sample_rate)

    def _process_fn(self, inarg: Any) -> torch.Tensor:
        return self.resampler(inarg)


class CMVNProcessor(Processor):
    """Processor to apply CMVN"""

    def __init__(self) -> None:
        super().__init__()

    def _process_fn(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        spectrum: (T, D)
            T is the number of frames in the utterance
            D is the number of the spectrum bins.
        """
        _mean = torch.mean(spectrum, dim=0)
        _std = torch.std(spectrum, dim=0)
        return (spectrum - _mean) / _std


def process_feat_as_kaldi(raw_audios: List[Tuple[str, str]], f_scp: str, f_ark: str, processor: Processor, desc: str = ''):
    with kaldiio.WriteHelper(f'ark,scp:{f_ark},{f_scp}') as writer:
        for uid, _audio in tqdm(raw_audios, desc=desc):
            writer(uid, processor(_audio).numpy())


def prepare_kaldi_feat(
        subsets: List[str],
        trans: Dict[str, List[Tuple[str, str]]],
        audios: Dict[str, List[Tuple[str, str]]],
        num_mel_bins: int = 80,
        apply_cmvn: bool = False,
        sample_frequency: Optional[int] = None,
        speed_perturb: Optional[List[float]] = [],
        fmt_scp: str = "data/src/all_ark/{}.scp",
        fmt_trans: str = "data/src/{}/text",
        fmt_ark: str = "data/src/.arks/{}.ark"):

    subsets = list(set(subsets))
    for _set in subsets:
        assert _set in trans
        assert _set in audios

    if sample_frequency is None:
        sample_frequency = torchaudio.load(audios[subsets[0]][0][1])[1]

    fbank_processor = FBankProcessor(sample_frequency, num_mel_bins)
    if apply_cmvn:
        fbank_processor = fbank_processor.append(CMVNProcessor())
    audio2fbank = ReadProcessor().append(fbank_processor)
    process_fn = process_feat_as_kaldi

    for _set in subsets:
        f_trans = fmt_trans.format(_set)
        f_scp = fmt_scp.format(_set)
        f_ark = fmt_ark.format(_set)
        os.makedirs(os.path.dirname(f_trans), exist_ok=True)
        os.makedirs(os.path.dirname(f_scp), exist_ok=True)
        os.makedirs(os.path.dirname(f_ark), exist_ok=True)

        # write transcript
        if os.path.isfile(f_trans):
            sys.stderr.write(
                f"warning: transcript {f_trans} exists, skip.\n")
        else:
            with open(f_trans, 'w') as fo:
                for uid, utt in trans[_set]:
                    fo.write(f"{uid}\t{utt}\n")

        # write feats
        if os.path.isfile(f_scp):
            sys.stderr.write(
                f"warning: scp file {f_scp} exists, skip.\n")
        else:
            process_fn(audios[_set], f_scp, f_ark, audio2fbank, desc=_set)

    for _factor in speed_perturb:
        if _factor == 1.0:
            continue
        sp_processor = SpeedPerturbationProcessor(
            _factor).append(fbank_processor)
        for _set in subsets:
            f_trans = fmt_trans.format(f"{_set}-sp{_factor}")
            f_scp = fmt_scp.format(f"{_set}-sp{_factor}")
            f_ark = fmt_ark.format(f"{_set}-sp{_factor}")
            os.makedirs(os.path.dirname(f_trans), exist_ok=True)
            os.makedirs(os.path.dirname(f_scp), exist_ok=True)
            os.makedirs(os.path.dirname(f_ark), exist_ok=True)
            # write trans
            if os.path.isfile(f_trans):
                sys.stderr.write(
                    f"warning: transcript {f_trans} exists, skip.\n")
            else:
                with open(f_trans, 'w') as fo:
                    for uid, utt in trans[_set]:
                        fo.write(f"{uid}#sp{_factor}\t{utt}\n")
            # write feats
            if os.path.isfile(f_scp):
                sys.stderr.write(
                    f"warning: scp file {f_scp} exists, skip.\n")
            else:
                process_fn(audios[_set], f_scp, f_ark,
                           sp_processor, desc=f"{_set} sp {_factor}")
