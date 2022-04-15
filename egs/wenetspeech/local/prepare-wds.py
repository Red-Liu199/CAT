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
        d_out: str,      # output directory
        fmt: str = "data-%05d.tar",       # tar file name format
        filter_group: List[str] = None,
        iszh: bool = False):
    """Parsing audio feature and text label into pickle file.

    Args:
        f_scps   (str, list): Kaldi-like-style .scp file(s).
        f_labels (str, list): Pure text file(s) include utterance id and sentence labels. Split by space.
        f_out   (str): Ouput pickle file location.
        filter (str, optional): identifier for filtering out seqs with unqualified length. 
            such as '100:2000' means remove those whose length is shorter than 100 or longer than 2000. Default: None
    """

    os.makedirs(d_out, exist_ok=True)

    if isinstance(f_scps, str):
        f_scps = [f_scps]
    if isinstance(f_labels, str):
        f_labels = [f_labels]

    # initialize filter bounds
    if filter_group is None:
        filter_group = [':']

    filter_bound = []   # type: List[Tuple[int, int]]
    for filt in filter_group:
        assert ':' in filt, f"parsingData: invalid filter format {filt}"
        l_bound, u_bound = (i for i in filt.split(':'))
        l_bound = 1 if l_bound == '' else int(l_bound)
        u_bound = 2**32-1 if u_bound == '' else int(u_bound)
        filter_bound.append((l_bound, u_bound))
    # it's your duty to ensure the intervals are not overlap
    filter_bound = sorted(filter_bound, key=lambda item: item[0])
    del filter_group
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
    if len(filter_bound) > 1:
        sinks = []
        for l, u in filter_bound:
            suffix = '' if u == (2**32-1) else str(u)
            subdir = os.path.join(d_out, f"{l}_{u}")
            os.makedirs(subdir, exist_ok=True)
            sinks.append(
                wds.ShardWriter(os.path.join(subdir, fmt), maxcount=2000)
            )
    else:
        sinks = [wds.ShardWriter(os.path.join(d_out, fmt), maxcount=2000)]

    for n, _f_scp in enumerate(f_scps):
        with open(_f_scp, 'r') as fi_scp:
            for line in tqdm(fi_scp, total=num_utts_individual[n]):
                key, loc_ark = line.split()
                tag = label_dict[key]
                feature = kaldiio.load_mat(
                    loc_ark, fd_dict=f_opened)   # type:np.ndarray

                L = feature.shape[0]
                for (l, u), _sink in zip(filter_bound, sinks):
                    if l <= L and L < u:
                        _sink.write({
                            '__key__': key,
                            "mat.npy": np.asarray(feature, dtype=np.float32),
                            "label.txt": tag
                        })
                        break

    for f in f_opened.values():
        f.close()

    for s in sinks:
        s.close()


if __name__ == "__main__":
    with open('data/.CATDATA.info', 'r') as fi:
        srcdata = json.load(fi)

    for subset in ['dev']:
        parsingData(
            f_scps=srcdata[subset]['scp'],
            f_labels=srcdata[subset]['trans'],
            d_out="./debug-wds",
            fmt=f"wenet-{subset}-%05d.tar",
            filter_group=["10:1500"],
            iszh=True
        )
