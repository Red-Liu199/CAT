import argparse
import pickle
import os
import string
import random
from multiprocessing import Pool
from typing import Union, Tuple


def randName(L: int) -> str:
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(L))


# def text2bin(args: argparse.Namespace, pool_id: int = 0, idx_beg: int = 0, idx_end: int = -1) -> Tuple[str, int]:
def text2bin(arguments: Tuple[argparse.Namespace, int, int, int]) -> Tuple[str, int]:

    args, pool_id, idx_beg, idx_end = arguments
    norm_len = args.concat
    binfile = '{}.{}.tmp'.format(args.outbin, randName(8))
    if idx_end == -1:
        idx_end = float('inf')

    dataset = []
    postfix = []
    cnt_process = 0
    tot_line = 0
    with open(args.intext, 'r') as fi:
        with open(binfile, 'wb') as fo:
            for i, line in (enumerate(fi)):
                if i < idx_beg or i >= idx_end:
                    continue

                tot_line += 1
                if args.strip:
                    data = [int(x) for x in line.split()[1:]]
                else:
                    data = [int(x) for x in line.split()]

                if norm_len != -1:
                    data = postfix + data
                    while len(data) >= norm_len:
                        dataset.append(fo.tell())
                        pickle.dump(data[:norm_len], fo)
                        data = data[norm_len:]
                        cnt_process += 1
                    if len(data) > 0:
                        postfix = data
                        postfix.append(args.sos_id)
                else:
                    dataset.append(fo.tell())
                    pickle.dump(data, fo)

    if norm_len != -1:
        print("[{:2}] Concat by len {}, # {} -> {}".format(pool_id,
              norm_len, tot_line, cnt_process))
        return pool_id, binfile, cnt_process
    else:
        return pool_id, binfile, tot_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("intext", type=str,
                        help="Input text files (in id format).")
    parser.add_argument("outbin", type=str, help="Ouput file.")
    parser.add_argument("--nj", type=int, default=1,
                        help="Number of threads. Default: 1")
    parser.add_argument("--strip", action="store_true", default=False)
    parser.add_argument("--concat", type=int, default=-1,
                        help="Use concat mode instead valid mode with given length. Default: -1 (disable)")
    parser.add_argument("--sos_id", type=int, default=0,
                        help="Begin of sequence index, available in concat > 1. Default: 0")

    args = parser.parse_args()

    num_threads = args.nj
    if num_threads < 1:
        raise ValueError(f"# threads must be >= 1, instead: {num_threads}")

    if not os.path.isfile(args.intext):
        raise FileNotFoundError(f"{args.intext} does not exist!")

    if num_threads == 1:
        pool_args = [(args, 0, 0, -1)]
    else:
        num_lines = sum(1 for _ in open(args.intext, 'r'))
        interval = num_lines // num_threads
        indices = list(range(0, num_lines, interval))
        if indices[-1] != num_lines - 1:
            indices[-1] = num_lines-1

        pool_args = [(args, i, indices[i], indices[i+1])
                     for i in range(num_threads)]

    with Pool(processes=num_threads) as pool:
        binfiles = pool.map(text2bin, pool_args)

    print("> Sub-process done. Begin merging...")
    binfiles = sorted(binfiles, key=lambda item: item[0])

    randbin = '{}.{}'.format(args.outbin, randName(8))
    _seeks = []
    with open(randbin, 'wb') as fo:
        for _, file, N in binfiles:
            with open(file, 'rb') as fi:
                for n in range(N):
                    _seeks.append(fo.tell())
                    pickle.dump(pickle.load(fi), fo)
            os.remove(file)

    with open(args.outbin, 'wb') as fo:
        # save the file name of binary file
        pickle.dump(randbin, fo)
        # save the location information
        pickle.dump(_seeks, fo)

    print("> Merged: Index file {} --> binary file {}".format(args.outbin, randbin))
