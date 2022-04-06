"""
Test the usage of webdataset module for the further develop
"""

from cat.shared.data import CorpusDataset
from cat.shared import coreutils
from cat.utils.pipeline_asr import readfromjson

import os
import io
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from typing import Union, List, Tuple

import webdataset as wds


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

    def __call__(self, batch: List[Tuple[np.ndarray, np.ndarray]]):
        batch_sorted = sorted(
            batch, key=lambda item: item[0].shape[0], reverse=True)
        batch_sorted = [(torch.as_tensor(x), torch.as_tensor(y))
                        for x, y in batch_sorted]

        X, Y = list(zip(*batch_sorted))
        input_lengths = torch.LongTensor(
            [x.size(0) for x in X])  # type: torch.LongTensor
        xs = coreutils.pad_list(X)   # type: torch.Tensor

        if self.flatten_target:
            target = torch.cat(Y, dim=0)
        else:
            target = coreutils.pad_list(Y)

        return xs, input_lengths, target, torch.empty(1)


def test_tarwriter():
    prev_data = CorpusDataset('exp/template-char-ngram/lmbin/train.pkl')
    with wds.TarWriter("dest.tar") as sink:
        for i in tqdm(range(len(prev_data)), leave=False):
            if i >= 5:
                break
            in_seq, target = prev_data[i]
            sink.write({
                '__key__': f"utt-{i:08}",
                "input.npy": np.asarray(i, dtype=np.int32),  # in_seq.numpy(),
                "output.npy": np.asarray(i+1, dtype=np.int32)  # target.numpy()
            })

    dataset = wds.WebDataset('dest.tar').decode().to_tuple(
        "input.npy", "output.npy")

    dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
    for in_seq, target in dataloader:
        print(in_seq)


def test_shardwriter():
    # prev_data = CorpusDataset('exp/template-char-ngram/lmbin/train.pkl')
    # with wds.ShardWriter("dest-%04d.tar", maxcount=1000) as sink:
    #     for i in tqdm(range(len(prev_data)), leave=False):
    #         in_seq, target = prev_data[i]
    #         sink.write({
    #             '__key__': f"utt-{i:08}",
    #             "input.npy": in_seq.numpy(),
    #             "output.npy": target.numpy()
    #         })

    dataset = wds.WebDataset('dest-{0000..0036}.tar', shardshuffle=True).shuffle(10000).decode().to_tuple(
        "input.npy", "output.npy")

    dataloader = wds.WebLoader(
        dataset, num_workers=0, batch_size=2048, collate_fn=sortedPadCollateLM())
    for i, item in enumerate(dataloader):
        if i < 10:
            continue
        if i >= 12:
            break
        inseq, lx, target, ly = item
        print(lx)


def parsingData(
        f_scps: Union[List[str], str],
        f_labels: Union[List[str], str],
        f_out_fmt: str,
        filter: str = None,
        iszh: bool = False):
    """Parsing audio feature and text label into pickle file.

    Args:
        f_scps   (str, list): Kaldi-like-style .scp file(s).
        f_labels (str, list): Pure text file(s) include utterance id and sentence labels. Split by space.
        f_out   (str): Ouput pickle file location.
        filter (str, optional): identifier for filtering out seqs with unqualified length. 
            such as '100:2000' means remove those whose length is shorter than 100 or longer than 2000. Default: None
        iszh (bool, optional): whether is chinese-liked lang (charater-based)
    """
    import kaldiio
    import numpy as np
    from tqdm import tqdm

    os.makedirs(os.path.dirname(f_out_fmt), exist_ok=True)

    if isinstance(f_scps, str):
        f_scps = [f_scps]
    if isinstance(f_labels, str):
        f_labels = [f_labels]

    l_min = 1
    l_max = float('inf')
    if filter is not None:
        assert ':' in filter, f"parsingData: invalid filter format {filter}"
        l_bound, u_bound = (i for i in filter.split(':'))
        if l_bound != '':
            l_min = int(l_bound)
        if u_bound != '':
            l_max = int(u_bound)

    # process label files
    labels = []
    for _f_lb in f_labels:
        with open(_f_lb, 'r') as fi_label:
            labels += fi_label.readlines()
    labels = [l.split() for l in labels]
    num_label_lines = len(labels)

    if iszh:
        labels = {l[0]: ''.join(l[1:]) for l in labels}
    else:
        labels = {l[0]: ' '.join(l[1:]) for l in labels}

    num_utts = [sum(1 for _ in open(_f_scp, 'r')) for _f_scp in f_scps]
    total_utts = sum(num_utts)
    assert total_utts == num_label_lines, \
        "parsingData: f_scp and f_label should match on the # lines, " \
        f"instead {total_utts} != {len(labels)}"

    f_opened = {}
    with wds.ShardWriter(f_out_fmt, maxcount=2000) as sink:
        for n, _f_scp in enumerate(f_scps):
            with open(_f_scp, 'r') as fi_scp:
                for line in tqdm(fi_scp, total=num_utts[n]):
                    key, loc_ark = line.split()
                    tag = labels[key]
                    feature = kaldiio.load_mat(
                        loc_ark, fd_dict=f_opened)   # type:np.ndarray

                    if feature.shape[0] < l_min or feature.shape[0] > l_max:
                        continue

                    sink.write({
                        '__key__': key,
                        "mat.npy": np.asarray(feature, dtype=np.float32),
                        "label.txt": tag
                    })

    for f in f_opened.values():
        f.close()


def trans_kaldi2tar(f_scp, f_text, fmt_dest):

    # train set
    parsingData(
        f_scp,
        f_text,
        fmt_dest,
        filter="10:1500",
        iszh=True)


def trans_old2tar(src_data: str):

    with open(src_data, 'rb') as fi:
        f_data = os.path.join(os.path.dirname(
            src_data), pickle.load(fi))  # type: str
        offsets = pickle.load(fi)

    with wds.ShardWriter("wenet-m-dev/wenet_m_dev-%04d.tar", maxcount=100) as sink, open(f_data, 'rb') as fi:
        for i in tqdm(range(len(offsets)), leave=False):
            fi.seek(offsets[i], 0)
            mat = np.load(fi)
            label = np.load(fi)
            sink.write({
                '__key__': f"utt-{i:08}",
                "mat.npy": mat,
                "label.npy": label
            })


if __name__ == "__main__":
    # test_shardwriter()
    # trans_old2tar(
    #     '/home/zhenghh/workspace/Transducer-dev/egs/wenetspeech/exp/rnnt-v1/pkl/dev.pkl')

    srcdata = readfromjson('data/.CATDATA.info')

    trans_kaldi2tar(
        f_scp=srcdata["tasi-reset-train-clean"]['scp'],
        f_text=srcdata["tasi-reset-train-clean"]['trans'],
        fmt_dest="/mnt/nvme_workspace/zhenghh/tasi-train-clean/tasi-%05d.tar",
    )
    trans_kaldi2tar(
        f_scp=srcdata["tasi-reset-train-noised"]['scp'],
        f_text=srcdata["tasi-reset-train-noised"]['trans'],
        fmt_dest="/mnt/nvme_workspace/zhenghh/tasi-train-noised/tasi-%05d.tar",
    )
