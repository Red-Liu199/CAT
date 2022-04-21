"""
Implementation of tokenizer
"""
import os
import io
import pickle
import sentencepiece as sp
import jieba

from collections import OrderedDict
from typing import List, Union, Iterable, Optional, Dict
from ..shared.coreutils import randstr


def gen_cache_path() -> str:
    return os.path.join('/tmp', randstr())


def file2bin(f_text: str) -> bytes:
    assert os.path.isfile(f_text), f"no such file: '{f_text}'"
    with open(f_text, 'rb') as fi:
        data = fi.read()
    return data


def bin2file(bindata: bytes, f_dest: Optional[str] = None) -> str:
    if f_dest is None:
        f_dest = gen_cache_path()
    with open(f_dest, 'wb') as fo:
        fo.write(bindata)
    return f_dest


class AbsTokenizer:
    def encode(self, strings: Union[str, Iterable[str]]) -> Union[List[int], List[List[int]]]:
        """Encode string into index."""
        raise NotImplementedError

    def decode(self, idx_tokens: Union[List[int], List[List[int]]]) -> Union[str, Iterable[str]]:
        """Decode index to string."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        raise NotImplementedError

    def dump_vocab(self, fileio: Optional[Union[str, "io.TextIOWrapper"]] = None) -> Union[None, Dict[int, str]]:
        """Dump vocabulary into a fileobject or return as dictionary"""
        vocab = self._vocab_to_dict()
        if fileio is None:
            return vocab
        elif isinstance(fileio, str):
            with open(fileio, 'w') as fo:
                for k, v in vocab.items():
                    fo.write(f"{v} {k}\n")
        elif isinstance(fileio, io.TextIOWrapper):
            for k, v in vocab.items():
                fileio.write(f"{v} {k}\n")
        else:
            raise ValueError(
                f"Unsupport file object of type: {type(fileio)}, expoected one of: str, TextIOWrapper")
        return None

    def _vocab_to_dict(self) -> Dict[int, str]:
        raise NotImplementedError

    def state_dict(self) -> OrderedDict:
        """Serialize tokenzier to dict object."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict):
        """Load tokenizer from serialized object"""
        raise NotImplementedError

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state: dict):
        self.load_state_dict(state)


# FIXME (huahuan): I'm not quite familiar with jieba, so there might be inappropriate processing


class JiebaTokenizer(AbsTokenizer):
    def __init__(self, userdict: Optional[Union[str, bytes]] = None, bos_id: int = 0, lazy_init: bool = False) -> None:
        super().__init__()
        if lazy_init:
            return
        self._tokenizer = jieba.Tokenizer()
        if userdict is None:
            self._tokenizer.initialize()
            self._vocabulary = list(self._tokenizer.FREQ.items())
            self.byte_dict = None
        else:
            if isinstance(userdict, str):
                assert os.path.isfile(
                    userdict), f"{userdict} is not a valid file."
                self._tokenizer.set_dictionary(userdict)
                self.byte_dict = file2bin(userdict)
                self._tokenizer.initialize()
            elif isinstance(userdict, bytes):
                self.byte_dict = userdict
                cachefile = bin2file(userdict)
                self._tokenizer.set_dictionary(cachefile)
                self._tokenizer.initialize()
                os.remove(cachefile)
            else:
                raise ValueError(f"Unknown userdict type: {type(userdict)}")
            # we only load user custom word
            self._vocabulary = [(w, None) for w, freq in self._tokenizer.FREQ.items(
            ) if freq > 0]  # type: Dict[str, int]

        if bos_id == 1:
            unk_id = 0
        else:
            unk_id = 1
        if bos_id == -1:
            bos_id = len(self._vocabulary) + 1
        assert bos_id < len(self._vocabulary) + 2

        self._vocabulary = OrderedDict(
            [('<s>', bos_id), ('<unk>', unk_id)] + self._vocabulary)
        self._reverse_vocab = tuple(self._vocabulary.keys())    # type: tuple
        for idx, w in enumerate(self._vocabulary):
            self._vocabulary[w] = idx

    def _enc(self, s: str) -> List[int]:
        cut_words = self._tokenizer.cut(s.replace(' ', ''), HMM=False)
        rt_indices = []     # type: List[int]
        for w in cut_words:
            if w not in self._vocabulary:
                w = "<unk>"
            rt_indices.append(self._vocabulary[w])
        return rt_indices

    def encode(self, strings: Union[str, Iterable[str]]) -> Union[List[int], List[List[int]]]:
        """Encode string to indices

        NOTE: since chinese language requires segmentation, so 
            tokenizer.decode(tokenizer.encode(string)) == string may not be satisfied.
        """
        if isinstance(strings, str):
            return self._enc(strings)
        try:
            iterator = iter(strings)
        except TypeError:
            raise RuntimeError(
                f"{self.__class__.__name__}.encode: input is neither str nor iterable.")

        cut_words = []
        for s in strings:
            cut_words.append(self._enc(s))
        return cut_words

    def decode(self, idx_tokens: Union[Iterable[int], Iterable[Iterable[int]]], seperator: str = '') -> Union[str, Iterable[str]]:
        try:
            iterator = iter(idx_tokens)
        except TypeError:
            raise RuntimeError(
                f"{self.__class__.__name__}.decode: input is not iterable.")

        if isinstance(next(iterator), int):
            iterator = iter(idx_tokens)
            return seperator.join([self._reverse_vocab[i] for i in iterator])

        iterator = iter(idx_tokens)
        try:
            sub_iterator = iter(next(iterator))
        except TypeError:
            raise RuntimeError(
                f"{self.__class__.__name__}.decode: element of input is neither int nor iterable.")

        out = []
        for item in idx_tokens:
            out.append(seperator.join([self._reverse_vocab[i] for i in item]))
        return out

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    def _vocab_to_dict(self) -> Dict[int, str]:
        return {idx: self._reverse_vocab[idx] for idx in range(self.vocab_size)}

    def state_dict(self) -> OrderedDict:
        return OrderedDict([
            ('vocab', self._vocabulary),
            ('reverse-vocab', self._reverse_vocab),
            ('dict-data', self.byte_dict)
        ])

    def load_state_dict(self, state_dict: OrderedDict):
        assert 'vocab' in state_dict
        assert 'reverse-vocab' in state_dict
        assert 'dict-data' in state_dict

        self._vocabulary = state_dict['vocab']
        self._reverse_vocab = state_dict['reverse-vocab']
        self.byte_dict = state_dict['dict-data']
        self._tokenizer = jieba.Tokenizer()
        if self.byte_dict is not None:
            cachefile = bin2file(self.byte_dict)
            self._tokenizer.set_dictionary(cachefile)
            self._tokenizer.initialize()
            os.remove(cachefile)
        else:
            self._tokenizer.initialize()


class SentencePieceTokenizer(AbsTokenizer):
    """SentencePiece tokenizer wrapper."""

    def __init__(self, spmodel: Optional[Union[str, bytes]] = None, lazy_init: bool = False) -> None:
        super().__init__()
        if lazy_init:
            # lazy initialize
            return
        assert spmodel is not None
        if not os.path.isfile(spmodel):
            raise RuntimeError(
                f"{self.__class__.__name__}: sentencepiece model path \'{spmodel}\' is invalid.")
        self._tokenzier = sp.SentencePieceProcessor(model_file=spmodel)
        self.byte_model = file2bin(spmodel)

    def encode(self, strings: Union[str, Iterable[str]]) -> Union[List[int], List[List[int]]]:
        return self._tokenzier.Encode(strings)

    def decode(self, idx_tokens: Union[List[int], List[List[int]]]) -> Union[str, Iterable[str]]:
        return self._tokenzier.Decode(idx_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenzier.vocab_size()

    def _vocab_to_dict(self) -> Dict[int, str]:
        out = {}
        for idx in range(self.vocab_size):
            out[idx] = self._tokenzier.IdToPiece(idx)
        return out

    def state_dict(self) -> OrderedDict:
        return OrderedDict([
            ('model-data', self.byte_model)
        ])

    def load_state_dict(self, state_dict: OrderedDict):
        assert 'model-data' in state_dict
        self.byte_model = state_dict['model-data']
        if self.byte_model is not None:
            cachefile = bin2file(self.byte_model)
            self._tokenzier = sp.SentencePieceProcessor(model_file=cachefile)
            os.remove(cachefile)


def save(obj: AbsTokenizer, target: str):
    """Save Tokenizer object at target location."""
    assert isinstance(
        obj, AbsTokenizer), "tokenizer.save: input object is not AbsTokenizer instance."
    with open(target, 'wb') as fo:
        pickle.dump(obj, fo)


def load(src: str) -> AbsTokenizer:
    assert os.path.isfile(src)
    with open(src, 'rb') as fi:
        tokenizer = pickle.load(fi)

    assert isinstance(
        tokenizer, AbsTokenizer), "tokenizer.load: loaded object is not AbsTokenizer instance."
    return tokenizer
