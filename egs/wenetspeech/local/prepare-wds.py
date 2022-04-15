"""
Test the usage of webdataset module for the further develop
"""

import os
import json
import kaldiio
import numpy as np
from tqdm import tqdm

from typing import Union, List, Tuple, Dict

import webdataset as wds


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
    """

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

    labels = [l.split(maxsplit=1)
              for l in labels]      # type: List[Tuple[str, str]]

    num_labels = len(labels)
    if iszh:
        # remove spaces for Asian lang
        label_dict = {}
        for i in range(num_labels):
            uid, utt = labels[i]
            label_dict[uid] = utt.replace(' ', '')
    else:
        label_dict = {uid: utt for uid, utt in labels}   # type: Dict[str, str]
    del labels

    num_utts_individual = [sum(1 for _ in open(_f_scp, 'r'))
                           for _f_scp in f_scps]
    num_utts = sum(num_utts_individual)
    assert num_utts == num_labels, \
        "parsingData: f_scp and f_label should match on the # lines, " \
        f"instead {num_utts} != {num_labels}"

    f_opened = {}
    with wds.ShardWriter(f_out_fmt, maxcount=2000) as sink:
        for n, _f_scp in enumerate(f_scps):
            with open(_f_scp, 'r') as fi_scp:
                for line in tqdm(fi_scp, total=num_utts_individual[n]):
                    key, loc_ark = line.split()
                    tag = label_dict[key]
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


if __name__ == "__main__":
    with open('data/.CATDATA.info', 'r') as fi:
        srcdata = json.load(fi)

    trans_kaldi2tar(
        f_scp=srcdata["train_l"]['scp'],
        f_text=srcdata["train_l"]['trans'],
        fmt_dest="./wenet-train-l-len1201_1500/wenetspeech-%05d.tar",
    )
