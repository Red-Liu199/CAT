"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
"""

import kaldi_io
import h5py
import utils
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, h5py_path):
        self.h5py_path = h5py_path
        hdf5_file = h5py.File(h5py_path, 'r')
        self.keys = hdf5_file.keys()
        hdf5_file.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        hdf5_file = h5py.File(self.h5py_path, 'r')
        dataset = hdf5_file[self.keys[idx]]
        mat = dataset.value
        label = dataset.attrs['label']
        weight = dataset.attrs['weight']
        hdf5_file.close()
        return torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)


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
              [torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key, feature_path, label, weight = self.dataset[idx]
        mat = np.array(kaldi_io.read_mat(feature_path))
        return torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)


class SpeechDatasetMemPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.data_batch = []

        for data in self.dataset:
            key, feature_path, label, weight = data
            mat = np.array(kaldi_io.read_mat(feature_path))
            self.data_batch.append(
                [torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])

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
        mat = np.array(kaldi_io.read_mat(feature_path))
        return key, torch.FloatTensor(mat), torch.LongTensor([mat.shape[0]])


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

        mats = utils.pad_list([x[0] for x in batch_sorted])

        labels = torch.cat([x[1] for x in batch_sorted])

        input_lengths = torch.LongTensor([x[3] for x in batch_sorted])

        label_lengths = torch.IntTensor([x[1].size(0) for x in batch_sorted])

        weights = torch.cat([x[2] for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths, weights


class sortedPadCollateTransducer():
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

        mats = utils.pad_list([x[0] for x in batch_sorted])

        labels = utils.pad_list([x[1] for x in batch_sorted]).to(torch.long)

        input_lengths = torch.LongTensor([x[3] for x in batch_sorted])

        label_lengths = torch.LongTensor([x[1].size(0) for x in batch_sorted])

        weights = torch.cat([x[2] for x in batch_sorted])

        ########## DEBUG CODE ###########
        # print(type(mats), mats.size())
        # print(type(input_lengths), input_lengths.size())
        # print(type(labels), labels.size())
        # print(type(label_lengths), label_lengths.size())
        # exit(1)
        #################################

        return mats, input_lengths, labels, label_lengths, weights
