'''
Author: Huahuan Zheng
This script used to compute WER of setences.
'''

import jiwer
import argparse
import os
import sys
import re
from multiprocessing import Pool
from typing import List, Tuple


def multi_run_wrapper(args):
   return WER(*args)


def WER(l_gt: List[str], l_hy: List[str]) -> Tuple[int, int, int, int]:
    measures = jiwer.compute_measures(l_gt, l_hy)
    return measures['substitutions'], measures['deletions'], measures['insertions'], measures['hits']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gt", type=str, help="Ground truth sequences.")
    parser.add_argument("hy", type=str, help="Hypothesis of sequences.")
    parser.add_argument("--stripid", action="store_true", default=False,
                        help="Tell whether the sequence start with a id or not. Default: False")
    args = parser.parse_args()

    assert os.path.isfile(args.gt)
    assert os.path.isfile(args.hy)
    ground_truth = args.gt  # type:str
    hypothesis = args.hy  # type:str

    with open(ground_truth, 'r') as f_gt:
        l_gt = f_gt.readlines()
        with open(hypothesis, 'r') as f_hy:
            l_hy = f_hy.readlines()

    num_lines = len(l_gt)
    assert num_lines == len(l_hy)

    pattern = re.compile(r' {2,}')
    l_gt = [pattern.sub(' ', x) for x in l_gt]
    l_gt = [x.strip('\n ') for x in l_gt]
    l_hy = [x.strip('\n ') for x in l_hy]

    if args.stripid:
        l_gt = [' '.join(x.split()[1:]) for x in l_gt]
        l_hy = [' '.join(x.split()[1:]) for x in l_hy]

    num_threads = int(os.cpu_count())

    interval = num_lines // num_threads
    indices = [interval * i for i in range(num_threads+1)]
    if indices[-1] != num_lines:
        indices[-1] = num_lines
    pool_args = [(l_gt[indices[i]:indices[i+1]], l_hy[indices[i]:indices[i+1]])
                 for i in range(num_threads)]

    with Pool(processes=num_threads) as pool:
        gathered_measures = pool.map(multi_run_wrapper, pool_args)

    _sub, _del, _ins, _hits = 0, 0, 0, 0
    for p_sub, p_del, p_ins, p_hits in gathered_measures:
        _sub += p_sub
        _del += p_del
        _ins += p_ins
        _hits += p_hits

    _err = _sub + _del + _ins
    _sum = _hits + _sub + _del
    _wer = _err / _sum

    # format: %WER 4.50 [ 2367 / 52576, 308 ins, 157 del, 1902 sub ]
    pretty_str = f"%WER {_wer*100:.2f} [{_err} / {_sum}, {_ins} ins, {_del} del, {_sub} sub ]"

    sys.stdout.write(pretty_str+'\n')
