'''
Author: Huahuan Zheng
This script used to compute WER of setences.
'''

import jiwer
import argparse
import os
import sys
import re
import pickle
from multiprocessing import Pool
from typing import List, Tuple, Callable, Union, Dict


class Processor():
    def __init__(self) -> None:
        self._process = []

    def append(self, new_processing: Callable[[str], str]):
        self._process.append(new_processing)
        pass

    def __call__(self, seqs: Union[List[str], str]) -> Union[List[str], str]:

        if isinstance(seqs, str):
            for processing in self._process:
                seqs = processing(seqs)
            return seqs
        else:
            o_seq = seqs
            for processing in self._process:
                o_seq = [processing(s) for s in o_seq]
            return o_seq


def WER(l_gt: List[str], l_hy: List[str]) -> Tuple[int, int, int, int]:
    measures = jiwer.compute_measures(l_gt, l_hy)
    return measures['substitutions'], measures['deletions'], measures['insertions'], measures['hits']


def oracleWER(l_gt: List[Tuple[str, str]], l_hy: List[Tuple[str, List[str]]]) -> Tuple[int, int, int, int]:
    """Computer oracle WER.

    Take first col of l_gt as key
    """

    l_hy = {key: nbest for key, nbest in l_hy}
    _sub, _del, _ins, _hit = 0, 0, 0, 0
    for key, g_s in l_gt:
        candidates = l_hy[key]
        best_wer = float('inf')
        best_measure = {}

        for can_seq in candidates:
            part_ith_measure = jiwer.compute_measures(g_s, can_seq)
            if part_ith_measure['wer'] < best_wer:
                best_wer = part_ith_measure['wer']
                best_measure = part_ith_measure

        _sub += best_measure['substitutions']
        _del += best_measure['deletions']
        _ins += best_measure['insertions']
        _hit += best_measure['hits']

    return _sub, _del, _ins, _hit


def run_wer_wrapper(args):
    return WER(*args)


def run_oracle_wer_wrapper(args):
    return oracleWER(*args)


def main(args: argparse.Namespace):

    ground_truth = args.gt  # type:str
    hypothesis = args.hy  # type:str
    assert os.path.isfile(ground_truth), ground_truth
    assert os.path.isfile(hypothesis), hypothesis

    with open(ground_truth, 'r') as f_gt:
        l_gt = f_gt.readlines()

    if args.oracle:
        # force to maintain ids
        args.stripid = False
        with open(hypothesis, 'rb') as f_hy:
            # type: Dict[str, List[Tuple[float, str]]]
            l_hy = pickle.load(f_hy)
        l_hy = [(key, nbest) for key, nbest in l_hy.items()]
    else:
        with open(hypothesis, 'r') as f_hy:
            l_hy = f_hy.readlines()

    num_lines = len(l_gt)
    assert num_lines == len(l_hy)

    # Pre-processing
    processor = Processor()

    # rm consecutive spaces
    pattern = re.compile(r' {2,}')
    processor.append(lambda s: pattern.sub(' ', s))

    # rm the '\n' and the last space
    processor.append(lambda s: s.strip('\n '))

    if args.cer:
        # rm space then split by char
        pattern = re.compile(r'\s+')
        processor.append(lambda s: pattern.sub('', s))
        processor.append(lambda s: ' '.join(list(s)))

    if args.oracle:
        for i, hypo in enumerate(l_hy):
            key, nbest = hypo
            seqs = processor([s for _, s in nbest])
            l_hy[i] = (key, seqs)

        for i, s in enumerate(l_gt):
            sl = s.split(' ')
            key, g_s = sl[0], ' '.join(sl[1:])
            l_gt[i] = (key, processor(g_s))

        l_hy = sorted(l_hy, key=lambda item: item[0])
        l_gt = sorted(l_gt, key=lambda item: item[0])
    elif args.stripid:
        for i, s in enumerate(l_gt):
            sl = s.split()
            key, g_s = sl[0], ' '.join(sl[1:])
            l_gt[i] = (key, processor(g_s))

        for i, s in enumerate(l_hy):
            sl = s.split()
            key, g_s = sl[0], ' '.join(sl[1:])
            l_hy[i] = (key, processor(g_s))

        l_hy = sorted(l_hy, key=lambda item: item[0])
        l_gt = sorted(l_gt, key=lambda item: item[0])
        l_hy = [seq for _, seq in l_hy]
        l_gt = [seq for _, seq in l_gt]
    else:
        l_hy = processor(l_hy)
        l_gt = processor(l_gt)

    # multi-processing compute
    num_threads = int(os.cpu_count())

    interval = num_lines // num_threads
    indices = [interval * i for i in range(num_threads+1)]
    if indices[-1] != num_lines:
        indices[-1] = num_lines
    pool_args = [(l_gt[indices[i]:indices[i+1]], l_hy[indices[i]:indices[i+1]])
                 for i in range(num_threads)]

    with Pool(processes=num_threads) as pool:
        if args.oracle:
            gathered_measures = pool.map(run_oracle_wer_wrapper, pool_args)
        else:
            gathered_measures = pool.map(run_wer_wrapper, pool_args)

    # gather sub-processes results
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
    prefix = 'WER' if not args.cer else 'CER'
    pretty_str = f"%{prefix} {_wer*100:.2f} [{_err} / {_sum}, {_ins} ins, {_del} del, {_sub} sub ]"

    sys.stdout.write(pretty_str+'\n')


def WERParser():
    parser = argparse.ArgumentParser('Compute WER/CER')
    parser.add_argument("gt", type=str, help="Ground truth sequences.")
    parser.add_argument("hy", type=str, help="Hypothesis of sequences.")
    parser.add_argument("--stripid", action="store_true", default=False,
                        help="Tell whether the sequence start with a id or not. When --oracle, this will be disable. Default: False")
    parser.add_argument("--cer", action="store_true", default=False,
                        help="Compute CER. Default: False")
    parser.add_argument("--oracle", action="store_true", default=False,
                        help="Compute Oracle WER/CER. This requires the `hy` to be N-best list instead of text. Default: False")
    return parser


if __name__ == "__main__":
    parser = WERParser()
    args = parser.parse_args()

    main(args)
