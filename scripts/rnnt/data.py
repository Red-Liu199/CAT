"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
"""

import os
import kaldiio
import h5py
import coreutils
import pickle
import math
from multiprocessing import Pool
from typing import Tuple, Sequence, List, Optional, Union

import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class FeatureReader:
    def __init__(self) -> None:
        self._opened_fd = {}

    def __call__(self, arkname: str):
        return kaldiio.load_mat(arkname, fd_dict=self._opened_fd)

    def __del__(self):
        for f in self._opened_fd.values():
            f.close()
        del self._opened_fd


class SpeechDataset(Dataset):
    def __init__(self, h5py_path):
        self.h5py_path = h5py_path
        self.dataset = None
        hdf5_file = h5py.File(h5py_path, 'r')
        self.keys = list(hdf5_file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5py_path, 'r')

        dataset = self.dataset[self.keys[idx]]
        mat = dataset[:]
        label = dataset.attrs['label']

        return torch.tensor(mat, dtype=torch.float), torch.IntTensor(label)


class SpeechDatasetMem(Dataset):
    def __init__(self, h5py_path):
        hdf5_file = h5py.File(h5py_path, 'r')
        keys = hdf5_file.keys()
        self.data_batch = []
        for key in keys:
          dataset = hdf5_file[key]
          mat = dataset[()]
          label = dataset.attrs['label']
          self.data_batch.append(
              [torch.tensor(mat, dtype=torch.float), torch.IntTensor(label)])

        hdf5_file.close()
        print("read all data into memory")

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        return self.data_batch[idx]


class SpeechDatasetPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.freader = FeatureReader()

    def get_seq_len(self) -> List[int]:
        _ls = []
        for _, feature_path, _ in self.dataset:
            mat = self.freader(feature_path)
            _ls.append(mat.shape[0])

        '''
        Files are opened in the parent process, so we close them.
        In __getitem__ function, they would be created again. This avoids
        errors with dataloder num_worker >= 1.
        '''
        del self.freader
        self.freader = FeatureReader()
        return _ls

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, feature_path, label = self.dataset[idx]
        mat = self.freader(feature_path)
        return torch.tensor(mat, dtype=torch.float), torch.IntTensor(label)


class SpeechDatasetMemPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.data_batch = []
        freader = FeatureReader()

        for data in self.dataset:
            key, feature_path, label = data
            mat = freader(feature_path)
            self.data_batch.append(
                [torch.tensor(mat, dtype=torch.float), torch.IntTensor(label)])

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        return self.data_batch[idx]


class InferDataset(Dataset):
    def __init__(self, scp_path) -> None:
        super().__init__()
        with open(scp_path, 'r') as fi:
            lines = fi.readlines()
        self.dataset = [x.split() for x in lines]
        self.freader = FeatureReader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key, feature_path = self.dataset[index]
        mat = self.freader(feature_path)
        return key, torch.tensor(mat, dtype=torch.float), torch.LongTensor([mat.shape[0]])


class ScpDataset(Dataset):
    """
    Read data from scp file ranging [idx_beg, idx_end)
    """

    def __init__(self, scp_file: str) -> None:
        super().__init__()

        if not os.path.isfile(scp_file):
            raise FileNotFoundError(f"{scp_file} is not a valid file.")

        self._dataset = []
        with open(scp_file, 'r') as fi:
            for line in fi:
                self._dataset.append(line.split())

        self.freader = FeatureReader()

    def __len__(self) -> int:
        return len(self._dataset)

    def get_seq_len(self) -> List[int]:
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


class CorpusDataset(Dataset):
    def __init__(self, pickle_path: str) -> None:
        super().__init__()
        assert os.path.isfile(pickle_path)

        self.dataset = None
        with open(pickle_path, 'rb') as fi:
            self._pathbin = pickle.load(fi)
            self._seeks = pickle.load(fi)

    def get_seq_len(self) -> List[int]:
        _ls = []
        for i in range(len(self)):
            data = self[i]
            _ls.append(data.size(0))
        self.dataset.close()
        self.dataset = None
        return _ls

    def __len__(self):
        return len(self._seeks)

    def __getitem__(self, index: int) -> torch.LongTensor:
        if self.dataset is None:
            self.dataset = open(self._pathbin, 'rb')

        self.dataset.seek(self._seeks[index], 0)
        data = pickle.load(self.dataset)    # type: Sequence[int]
        return torch.LongTensor(data)


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

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

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

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

        labels = coreutils.pad_list(
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

        mats = coreutils.pad_list([feature for _, feature in batch])

        lengths = torch.LongTensor([feature.size(0) for _, feature in batch])

        return keys, mats, lengths


class sortedPadCollateLM():
    """Collect data into batch by desending order and add padding.

    Args:
        batch  : list of label
            label  : torch.LongTensor

    Return:
        (labels_with_bos, label_lengths, labels_with_eos, `torch.empty(1)`, `torch.empty(1)`)
    """

    def __call__(self, batch: Sequence[torch.LongTensor]):
        batches = [(label, label.size(0)) for label in batch]

        batch_sorted = sorted(batches, key=lambda item: item[1], reverse=True)

        xs = coreutils.pad_list([x[0] for x in batch_sorted]
                                )   # type: torch.Tensor
        # xs -> <s> + xs
        xs = torch.cat([xs.new_zeros(xs.size(0), 1), xs], dim=1)

        # labels -> labels + <s>
        labels = [torch.cat([x, x.new_zeros(1)]) for x, _ in batch_sorted]
        labels = torch.cat(labels)

        input_lengths = torch.LongTensor([l+1 for _, l in batch_sorted])

        label_lengths = torch.empty(1)

        return xs, input_lengths, labels, label_lengths


class BalancedDistributedSampler(DistributedSampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 global_batch_size: int,
                 length_norm: Optional[str] = None,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
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
        if rank == 0:
            seq_lens = dataset.get_seq_len()
        else:
            seq_lens = [0 for _ in range(len(self.dataset))]
        dist.broadcast_object_list(seq_lens)

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

        num_threads = min(int(os.cpu_count()) //
                          self.num_replicas, len(batched_indices))
        interval = len(batched_indices) // num_threads
        process_idx = [interval * i for i in range(num_threads+1)]
        if process_idx[-1] != len(batched_indices):
            process_idx[-1] = len(batched_indices)

        pool_args = [(batched_indices[process_idx[i]:process_idx[i+1]], self._lens, self._l_norm, self.num_replicas, i)
                     for i in range(num_threads)]
        with Pool(processes=num_threads) as pool:
            gathered_groups = pool.map(group_indices, pool_args)

        partial_indices = []
        for g, _ in sorted(gathered_groups, key=lambda i: i[1]):
            partial_indices += g
        partial_indices = [x[self.rank] for x in partial_indices]

        return iter(partial_indices)


def group_indices(args: Tuple[List[List[int]], List[int], Union[str, None], int, int]):
    idx_groups, global_ls, l_norm, n_replicas, p_id = args
    for k, g in enumerate(idx_groups):
        g_sorted = sorted(g, key=lambda i: global_ls[i], reverse=True)

        g_grouped = coreutils.group_by_lens(
            g_sorted, [global_ls[i] for i in g_sorted], n_replicas, l_norm)
        idx_groups[k] = g_grouped

    return idx_groups, p_id


class InferenceDistributedSampler(BalancedDistributedSampler):
    def __init__(self, dataset: torch.utils.data.Dataset, length_norm: Optional[str] = None) -> None:
        world_size = dist.get_world_size()
        super().__init__(dataset, world_size, length_norm=length_norm, shuffle=False)

    def __iter__(self):
        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # split samples to make it evenly divisible
        num_samples = len(self.dataset)
        res_size = num_samples % self.num_replicas
        if res_size == 0:
            res_indices = []
            indices = list(range(num_samples))
        else:
            indices = list(range(num_samples-res_size))
            res_indices = list(range(num_samples-res_size, num_samples))

        # Add implementation here
        partial_indices = []

        batches = sorted(indices, key=lambda i: self._lens[i], reverse=True)
        batches = coreutils.group_by_lens(
            batches, [self._lens[i] for i in batches],
            self.num_replicas, self._l_norm, False)

        partial_indices = batches[self.rank]

        if res_size > 0 and self.rank < res_size:
            partial_indices.append(res_indices[self.rank])
        return iter(partial_indices)
