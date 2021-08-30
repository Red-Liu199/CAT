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
from kaldiio import ReadHelper
from typing import Union, Tuple, Sequence

import torch
from torch.utils.data import Dataset


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
        weight = dataset.attrs['weight']

        return torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)


class SpeechDatasetMem(Dataset):
    def __init__(self, h5py_path):
        hdf5_file = h5py.File(h5py_path, 'r')
        keys = hdf5_file.keys()
        self.data_batch = []
        for key in keys:
          dataset = hdf5_file[key]
          mat = dataset[()]
          label = dataset.attrs['label']
          weight = dataset.attrs['weight']
          self.data_batch.append(
              [torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)])

        hdf5_file.close()
        print("read all data into memory")

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        return self.data_batch[idx]


class CorpusDataset(Dataset):
    def __init__(self, pickle_path: str) -> None:
        super().__init__()
        assert os.path.isfile(pickle_path)

        self.dataset = None
        with open(pickle_path, 'rb') as fi:
            self._pathbin = pickle.load(fi)
            self._seeks = pickle.load(fi)

    def __len__(self):
        return len(self._seeks)

    def __getitem__(self, index: int) -> torch.LongTensor:
        if self.dataset is None:
            self.dataset = open(self._pathbin, 'rb')

        self.dataset.seek(self._seeks[index], 0)
        data = pickle.load(self.dataset)    # type: Sequence[int]
        return torch.LongTensor(data)


class SpeechDatasetPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key, feature_path, label, weight = self.dataset[idx]
        mat = kaldiio.load_mat(feature_path)
        return torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)


class SpeechDatasetMemPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.data_batch = []

        for data in self.dataset:
            key, feature_path, label, weight = data
            mat = kaldiio.load_mat(feature_path)
            self.data_batch.append(
                [torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)])

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key, feature_path = self.dataset[index]
        mat = kaldiio.load_mat(feature_path)
        return key, torch.tensor(mat, dtype=torch.float), torch.LongTensor([mat.shape[0]])


class sortedPadCollate():
    def __call__(self, batch):
        """Collect data into batch by desending order and add padding.

        Args: 
            batch  : list of (mat, label, weight)
            mat    : torch.FloatTensor
            label  : torch.IntTensor
            weight : torch.FloatTensor

        Return: 
            (logits, input_lengths, labels, label_lengths, weights)
        """
        batches = [(mat, label, weight, mat.size(0))
                   for mat, label, weight in batch]
        batch_sorted = sorted(batches, key=lambda item: item[3], reverse=True)

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

        labels = torch.cat([x[1] for x in batch_sorted])

        input_lengths = torch.LongTensor([x[3] for x in batch_sorted])

        label_lengths = torch.IntTensor([x[1].size(0) for x in batch_sorted])

        weights = torch.cat([x[2] for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths, weights


class sortedPadCollateTransducer():
    """Collect data into batch by desending order and add padding.

    Args: 
        batch  : list of (mat, label, weight)
        mat    : torch.FloatTensor
        label  : torch.IntTensor
        weight : torch.FloatTensor

    Return: 
        (logits, input_lengths, labels, label_lengths, weights)
    """

    def __call__(self, batch):
        batches = [(mat, label, weight, mat.size(0))
                   for mat, label, weight in batch]
        batch_sorted = sorted(batches, key=lambda item: item[3], reverse=True)

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

        labels = coreutils.pad_list(
            [x[1] for x in batch_sorted]).to(torch.long)

        input_lengths = torch.LongTensor([x[3] for x in batch_sorted])

        label_lengths = torch.LongTensor([x[1].size(0) for x in batch_sorted])

        weights = torch.cat([x[2] for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths, weights


class ScpDataset(Dataset):
    """
    Read data from scp file ranging [idx_beg, idx_end)
    """

    def __init__(self, scp_file, idx_beg: int = 0, idx_end: int = -1) -> None:
        super().__init__()

        if not os.path.isfile(scp_file):
            raise FileNotFoundError(f"{scp_file} is not a valid file.")

        assert idx_beg >= 0 and idx_end >= -1

        if idx_end == -1:
            idx_end = float('inf')

        self._dataset = []
        idx = 0
        with ReadHelper('scp:'+scp_file) as reader:
            for key, mat in reader:
                if idx < idx_beg:
                    idx += 1
                    continue
                if idx >= idx_end:
                    break
                self._dataset.append(
                    [key, torch.tensor(mat, dtype=torch.float)])
                idx += 1

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor]:
        return self._dataset[index]


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

        weights = torch.empty(1)

        return xs, input_lengths, labels, label_lengths, weights
