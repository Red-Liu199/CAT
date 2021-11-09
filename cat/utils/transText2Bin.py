"""Convert text to pickle data

text could be in token index format or pure text (with tokenizer specified)
"""

import argparse
import pickle
import os
import uuid
import sentencepiece as spm
from multiprocessing import Pool
from typing import Union, Tuple


def text2bin(arguments: Tuple[argparse.Namespace, str, int, int]):

    args, binfile, idx_beg, idx_end = arguments
    if idx_end == -1:
        idx_end = float('inf')

    if args.spm is not None:
        spmodel = spm.SentencePieceProcessor(model_file=args.spm)
        processor = spmodel.encode
    else:
        def processor(line): return [int(x) for x in line.split()]

    dataset = []
    postfix = []
    cnt_process = 0
    tot_line = 0
    flag_norm = (args.concat != -1) or (args.truncate != -1)
    istruncate = (args.truncate != -1)
    if istruncate:
        norm_len = args.truncate
    else:
        norm_len = args.concat

    with open(args.intext, 'r') as fi:
        for i, line in (enumerate(fi)):
            if i < idx_beg or i >= idx_end:
                continue
            tot_line += 1
            l_data = processor(line)

            if flag_norm:
                data = postfix + l_data
                while len(data) >= norm_len:
                    dataset.append(data[:norm_len])
                    data = data[norm_len:]
                    cnt_process += 1
                if len(data) > 0:
                    if istruncate:
                        dataset.append(data)
                        cnt_process += 1
                    else:
                        postfix = data
                        postfix.append(args.bos_id)
            else:
                dataset.append(l_data)

    with open(binfile, 'wb') as fo:
        pickle.dump(dataset, fo)

    if flag_norm:
        if istruncate:
            print("Truncate by {}, # {} -> {}".format(
                args.truncate, tot_line, cnt_process))
        else:
            print("Concat by {}, # {} -> {}".format(
                norm_len, tot_line, cnt_process))


def main(args: argparse.Namespace):
    assert args.truncate == -1 or args.concat == - \
        1, "--concat is conflict with --truncate"

    num_threads = args.nj
    if num_threads < 1:
        raise ValueError(f"# threads must be >= 1, instead: {num_threads}")

    if not os.path.isfile(args.intext):
        raise FileNotFoundError(f"{args.intext} does not exist!")

    fmt = os.path.join('/tmp', str(uuid.uuid4())+'.{}.tmp')
    if num_threads == 1:
        pool_args = [(args, fmt.format(0), 0, -1)]
    else:
        num_lines = sum(1 for _ in open(args.intext, 'r'))
        interval = num_lines // num_threads
        indices = [interval * i for i in range(num_threads+1)]
        if indices[-1] != num_lines:
            indices[-1] = num_lines

        pool_args = [(args, fmt.format(i), indices[i], indices[i+1])
                     for i in range(num_threads)]

    with Pool(processes=num_threads) as pool:
        pool.map(text2bin, pool_args)

    print("> Sub-process done. Begin merging...")

    randbin = '{}.bin'.format(args.output)
    _seeks = []
    with open(randbin, 'wb') as fo:
        for i in range(num_threads):
            with open(fmt.format(i), 'rb') as fi:
                part_dataset = pickle.load(fi)

            for _data in part_dataset:
                _seeks.append(fo.tell())
                pickle.dump(_data, fo)
            os.remove(fmt.format(i))

    with open(args.output, 'wb') as fo:
        # save the file name of binary file
        pickle.dump(randbin, fo)
        # save the location information
        pickle.dump(_seeks, fo)

    print("> Merged: Index {} --> binary {}".format(args.output, randbin))


def TextProcessingParser():
    parser = argparse.ArgumentParser(
        'Convert pure text into pickle data with multi-processing')
    parser.add_argument("intext", type=str,
                        help="Input text files (in token id if no --spm, or text  with --spm).")
    parser.add_argument("output", type=str, help="Ouput file.")
    parser.add_argument("--spm", type=str,
                        help="Location of sentencepiece model.")
    parser.add_argument("--nj", type=int, default=1,
                        help="Number of threads. Default: 1")
    parser.add_argument("--concat", type=int, default=-1,
                        help="Use concat mode instead valid mode with given length. Default: -1 (disable)")
    parser.add_argument("--truncate", type=int, default=-1, metavar="trunc",
                        help="Truncate the seq longer than trunc and take res of it as new seq. Default: -1 (disable)")
    parser.add_argument("--bos_id", type=int, default=0,
                        help="Begin of sequence index, available in concat > 1. Default: 0")
    return parser


if __name__ == "__main__":
    parser = TextProcessingParser()
    args = parser.parse_args()
    main(args)
