# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Hongyu Xiang,
#         Keyu An,
#         Zheng Huahuan (maxwellzh@outlook.com)

"""Data loading module
"""

from . import coreutils as utils
from .tokenizer import AbsTokenizer

import io
import os
import kaldiio
import pickle
import math
import hashlib
import numpy as np
from collections import OrderedDict
from typing import Tuple, Sequence, List, Optional, Union, Dict

import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def get_sha256(file: str) -> str:
    '''Get sha256 has of a file.
    '''
    assert os.path.isfile(file), f"{file} not found."
    sha256_hash = hashlib.sha256()
    with open(file, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


class FeatureReader:
    def __init__(self) -> None:
        self._opened_fd = {}

    def __call__(self, arkname: str):
        return kaldiio.load_mat(arkname, fd_dict=self._opened_fd)

    def __del__(self):
        for f in self._opened_fd.values():
            f.close()
        del self._opened_fd


class AbsDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.f_path = path
        assert os.path.isfile(
            path), f"{self.__class__.__name__}: {path} is not a valid file."

    def impl_get_len(self):
        raise NotImplementedError

    def get_seq_len(self) -> List[int]:
        # try to find length info otherwise read from features
        f_linfo = self.f_path+'.linfo'
        if os.path.isfile(f_linfo):
            with open(f_linfo, 'rb') as fi:
                return pickle.load(fi)
        else:
            ls = self.impl_get_len()
            if not os.access(os.path.dirname(f_linfo), os.W_OK):
                print(
                    f"Cannot write length info into dir: {os.path.dirname(f_linfo)}. Would reload it at the next time")
            else:
                with open(f_linfo, 'wb') as fo:
                    pickle.dump(ls, fo)
            return ls


class IndexMappingDataset(AbsDataset):
    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)
        self.dataset = None
        with open(f_index, 'rb') as fi:
            self.f_data = os.path.join(os.path.dirname(
                f_index), pickle.load(fi))  # type: str
            if not os.path.isfile(self.f_data):
                raise FileNotFoundError(
                    f"\n{self.__class__.__name__}:\n"
                    f"From indexing file {f_index} mapping to {self.f_data}\n"
                    f"... but {self.f_data} is not found.")
            self.offsets = pickle.load(fi)

    def impl_get_len(self):
        _ls = np.empty(len(self), dtype=np.int64)
        for i in range(len(self)):
            ''' NOTE (huahuan): 
            suppose `__getitem__` method returns a tuple
            ... where the first item is the feature;
            ... if not the case, impl your custom `impl_get_len` method.
            '''
            x = self[i][0]
            _ls[i] = x.size(0)
        return _ls

    def __len__(self) -> int:
        return len(self.offsets)

    @staticmethod
    def _readbuffer(fileio: "io.BufferedReader"):
        raise NotImplementedError

    def __getitem__(self, index: int):
        if self.dataset is None:
            self.dataset = open(self.f_data, 'rb')
        self.dataset.seek(self.offsets[index], 0)
        # you should impl `_readbuffer` method of your derived class
        return self._readbuffer(self.dataset)


# NOTE (Huahuan):
#    deprecate old speech dataset for better CPU memory efficiency,
#    ... check https://pytorch.org/docs/stable/data.html#multi-process-data-loading
#    ... for why this happened.


class ModifiedSpeechDataset(IndexMappingDataset):
    """Speech dataset"""

    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)

    @staticmethod
    def _readbuffer(fileio: "io.BufferedReader"):
        mat = np.load(fileio)
        label = np.load(fileio)
        return torch.from_numpy(mat), torch.from_numpy(label)


class CorpusDataset(IndexMappingDataset):
    """LM corpus dataset"""

    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)

    @staticmethod
    def _readbuffer(fileio: "io.BufferedReader"):
        x, y = pickle.load(fileio)
        return torch.LongTensor(x), torch.LongTensor(y)


class ScpDataset(AbsDataset):
    """
    Read data from scp file ranging [idx_beg, idx_end)
    """

    def __init__(self, scp_file: str) -> None:
        super().__init__(scp_file)

        if not os.path.isfile(scp_file):
            raise FileNotFoundError(f"{scp_file} is not a valid file.")

        self._dataset = []
        with open(scp_file, 'r') as fi:
            for line in fi:
                self._dataset.append(line.split())

        self.freader = FeatureReader()

    def __len__(self) -> int:
        return len(self._dataset)

    def impl_get_len(self):
        _ls = []
        for _, fpath in self._dataset:
            mat = self.freader(fpath)
            _ls.append(mat.shape[0])

        del self.freader
        self.freader = FeatureReader()
        return _ls

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor]:
        key, mat_path = self._dataset[index]
        mat = self.freader(mat_path)
        return [key, torch.tensor(mat, dtype=torch.float)]


class NbestListDataset(AbsDataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        with open(self.f_path, 'rb') as fi:
            # type: Dict[str, Dict[int, Tuple[float, str]]]
            self._dataset = list(pickle.load(fi).items())

    def impl_get_len(self):
        return [sum([len(hyp) for _, hyp in hypos]) for _, hypos in self._dataset]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[List[str], List[float], List[str]]:
        # create new key = nbest id + '-' + original key,
        # so that we can get it back via new_key.split('-', maxsplit=1)
        keys, scores, trans = [], [], []
        okey = self._dataset[index][0]
        for nid, (_score, _trans) in self._dataset[index][1].items():
            keys.append(f"{nid}-{okey}")
            scores.append(_score)
            trans.append(_trans)
        return keys, scores, trans


class NbestListCollate():
    """Collator for N-best list file.
    The passing tokenizer should have method `encode` to convert text to indices.
    """

    def __init__(self, tokenizer: AbsTokenizer, bos_id: int = 0) -> None:
        self._tokenizer = tokenizer
        assert isinstance(
            bos_id, int) and bos_id >= 0, f"ValueError: bos_id={bos_id}"
        self.bos_id = bos_id

    def __call__(self, batches: List[Tuple[List[str], List[float], List[str]]]):
        """
        Args:
            batches : [([key, key, ...], [score1, score2, ...], ["hypo1", "hypo2",...]), ...], length B

        Returns:
            (keys, texts, scores, tokens)
            keys (List[str]): (B * N-best, )
            texts (List[str]): (B * N-best, )
            scores (torch.FloatTensor): (B * N-best, )
            tokens :
            {
                'input_ids' (torch.LongTensor): (B * N-best, L_max)
                'attention_mask' (torch.LongTensor, torch.BoolTensor): (B * N-best, L_max)
            }
        """
        keys, scores, trans = [], [], []
        for lk, ls, lt in batches:
            keys += lk
            scores += ls
            trans += lt

        ids = [[self.bos_id] +
               self._tokenizer.encode(seqs) for seqs in trans]
        token_ids = utils.pad_list(
            [torch.LongTensor(i) for i in ids])
        lens = torch.LongTensor([len(x) for x in ids])
        token_mask = torch.arange(
            lens.max())[None, :] >= lens[:, None]

        scores = torch.FloatTensor(scores)
        return keys, trans, scores, token_ids, token_mask


class sortedPadCollate():
    def __call__(self, batch):
        """Collect data into batch by desending order and add padding.

        Args:
            batch  : list of (mat, label)
            mat    : torch.FloatTensor
            label  : torch.IntTensor

        Return:
            (logits, input_lengths, labels, label_lengths)
        """
        batches = [(mat, label, mat.size(0))
                   for mat, label in batch]
        batch_sorted = sorted(batches, key=lambda item: item[2], reverse=True)

        mats = utils.pad_list([x[0] for x in batch_sorted])

        labels = torch.cat([x[1] for x in batch_sorted])

        input_lengths = torch.LongTensor([x[2] for x in batch_sorted])

        label_lengths = torch.IntTensor([x[1].size(0) for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths


class sortedPadCollateTransducer():
    """Collect data into batch by desending order and add padding.

    Args:
        batch  : list of (mat, label)
        mat    : torch.FloatTensor
        label  : torch.IntTensor

    Return:
        (logits, input_lengths, labels, label_lengths)
    """

    def __call__(self, batch):
        batches = [(mat, label, mat.size(0))
                   for mat, label in batch]
        batch_sorted = sorted(batches, key=lambda item: item[2], reverse=True)

        mats = utils.pad_list([x[0] for x in batch_sorted])

        labels = utils.pad_list(
            [x[1] for x in batch_sorted]).to(torch.long)

        input_lengths = torch.LongTensor([x[2] for x in batch_sorted])

        label_lengths = torch.LongTensor([x[1].size(0) for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths


class TestPadCollate():
    """Collect data into batch and add padding.

    Args:
        batch   : list of (key, feature)
        key     : str
        feature : torch.FloatTensor

    Return:
        (keys, logits, lengths)
    """

    def __call__(self, batch: Sequence[Tuple[str, torch.FloatTensor]]) -> Tuple[Sequence[str], torch.FloatTensor, torch.LongTensor]:

        keys = [key for key, _ in batch]

        mats = utils.pad_list([feature for _, feature in batch])

        lengths = torch.LongTensor([feature.size(0) for _, feature in batch])

        return keys, mats, lengths


class sortedPadCollateLM():
    """Collect data into batch by desending order and add padding.

    Args:
        batch  : [(labels, targets)]
            labels  : torch.LongTensor
            targets : torch.LongTensor

    Return:
        (labels, label_lengths, targets, `torch.empty(1)`)
    """

    def __init__(self, flatten_target: bool = True) -> None:
        self.flatten_target = flatten_target

    def __call__(self, batch: Tuple[List[torch.LongTensor], List[torch.LongTensor]]):
        batch_sorted = sorted(
            batch, key=lambda item: item[0].size(0), reverse=True)

        X, Y = list(zip(*batch_sorted))
        input_lengths = torch.LongTensor(
            [x.size(0) for x in X])  # type: torch.LongTensor
        xs = utils.pad_list(X)   # type: torch.Tensor

        if self.flatten_target:
            target = torch.cat(Y, dim=0)
        else:
            target = utils.pad_list(Y)

        return xs, input_lengths, target, torch.empty(1)


class BalancedDistributedSampler(DistributedSampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 global_batch_size: int,
                 length_norm: Optional[str] = None,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 local_rank: int = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)

        if global_batch_size < self.num_replicas or global_batch_size > len(self.dataset):
            raise RuntimeError(
                "Invalid global batch size: ", global_batch_size)

        if not hasattr(dataset, 'get_seq_len'):
            raise RuntimeError(
                f"{type(dataset)} has not implement Dataset.get_seq_len method, which is required for BalanceDistributedSampler.")

        # scan data length, this might take a while
        if rank is None:
            rank = dist.get_rank()
        if local_rank is None:
            # using 1 node
            local_rank = rank

        if local_rank == 0:
            # save length info into cache file
            dataset.get_seq_len()

        dist.barrier()

        # read from cached file
        seq_lens = dataset.get_seq_len()
        self._lens = seq_lens

        self.g_batch = int(global_batch_size)
        self._l_norm = length_norm

    def __iter__(self):
        # DistributedSampler.__iter__()
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Add implementation here
        batched_indices = [indices[idx_g_batch:idx_g_batch+self.g_batch]
                           for idx_g_batch in range(0, self.total_size, self.g_batch)]

        if len(batched_indices[-1]) < self.num_replicas:
            batched_indices.pop()

        partial_indices, _ = group_indices(
            (batched_indices, self._lens, self._l_norm, self.num_replicas, 0))

        local_indices = [x[self.rank] for x in partial_indices]
        return iter(local_indices)


def group_indices(args: Tuple[List[List[int]], List[int], Union[str, None], int, int]):
    idx_groups, global_ls, l_norm, n_replicas, p_id = args
    for k, g in enumerate(idx_groups):
        g_sorted = sorted(g, key=lambda i: global_ls[i], reverse=True)

        g_grouped = utils.group_by_lens(
            g_sorted, [global_ls[i] for i in g_sorted], n_replicas, l_norm)
        idx_groups[k] = g_grouped

    return idx_groups, p_id
