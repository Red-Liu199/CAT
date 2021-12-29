"""Convert text to pickle data

text could be in token index format or pure text (with tokenizer specified)

would create two files: (suppose --output=<f_text>)
    <f_text> and <f_text>.bin
    where <f_text> stores the location of <f_text>.bin, as well as 
    the data location given by fseek(). 

How to get the parsed data:
    with open('<f_text>', 'rb') as fi:
        f_bin = pickle.load(fi)
        f_seeks = pickle.load(fi)
    
    with open(f_bin, 'rb') as fi:
        # get the 5th element
        index = 5
        fi.seek(f_seeks[index], 0)
        data = pickle.load(fi)
"""

import argparse
import pickle
import os
import uuid
import sentencepiece as spm
from multiprocessing import Pool
from typing import Union, Tuple, List


def chunk(X: List[int], Y: List[int], chunk_size: int, drop_res: bool = True):
    assert len(X) == len(Y)
    lx = len(X)
    if drop_res:
        assert lx >= chunk_size
        res_size = lx % chunk_size
    else:
        res_size = 0

    for bound in range(0, lx-res_size, chunk_size):
        yield X[bound:bound+chunk_size], Y[bound:bound+chunk_size]


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
    cnt_process = 0
    tot_line = 0
    concat_lines = []   # type: List[int]
    with open(args.intext, 'r') as fi:
        for i, line in (enumerate(fi)):
            if i < idx_beg or i >= idx_end:
                continue
            tot_line += 1
            l_data = processor(line)

            if args.truncate != -1:
                # <s> + seq
                X = [args.bos_id]+l_data
                # seq + </s>
                Y = l_data+[args.eos_id]
                for x, y in chunk(X, Y, args.truncate, drop_res=False):
                    dataset.append((x, y))
                    cnt_process += 1
            elif args.concat != -1:
                concat_lines.append(args.bos_id)
                concat_lines += l_data
            else:
                # (<s> + seq, seq + </s>)
                dataset.append(
                    ([args.bos_id]+l_data, l_data+[args.eos_id]))

    if args.concat != -1:
        X = concat_lines
        Y = concat_lines[1:] + [args.bos_id]
        for x, y in chunk(X, Y, args.concat, drop_res=True):
            dataset.append((x, y))
            cnt_process += 1

    with open(binfile, 'wb') as fo:
        pickle.dump(dataset, fo)

    if args.truncate != -1:
        print("Truncate by {}, # {} -> {}".format(
            args.truncate, tot_line, cnt_process))
    elif args.concat != -1:
        print("Concat by {}, # {} -> {}".format(
            args.concat, tot_line, cnt_process))


def main(args: argparse.Namespace):
    assert args.truncate == -1 or args.concat == - \
        1, "--concat is conflict with --truncate"

    num_threads = args.nj
    if num_threads < 1:
        raise ValueError(f"# threads must be >= 1, instead: {num_threads}")

    if not os.path.isfile(args.intext):
        raise FileNotFoundError(f"{args.intext} does not exist!")

    if args.eos_id == -1:
        args.eos_id = args.bos_id

    if args.concat > 1 and (args.eos_id != args.bos_id):
        raise RuntimeError(
            f"--concat > 1 requires <bos> = <eos>, instead {args.bos_id} != {args.eos_id}")

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
        pickle.dump(os.path.abspath(randbin), fo)
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
                        help="Begin of sequence index, used when concat > 1. Default: 0")
    parser.add_argument("--eos_id", type=int, default=-1,
                        help="End of sequence index, used when concat > 1. Default: -1 (same as --bos_id)")
    return parser


if __name__ == "__main__":
    parser = TextProcessingParser()
    args = parser.parse_args()
    main(args)
