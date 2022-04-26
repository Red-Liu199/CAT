# Author: Huahuan Zheng (maxwellzh@outlook.com)

from cat.shared.data import CorpusDataset
from cat.shared import tokenizer as tknz

import os
import sys
import argparse
import uuid
from typing import *
from multiprocessing import Pool


def blocks(files, size=2**16):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def countlines(files: Union[str, List[str]]) -> int:
    if isinstance(files, str):
        files = [files]
    cnt = 0
    for _file in files:
        with open(_file, "r", encoding="utf-8", errors='ignore') as f:
            cnt += sum(bl.count("\n") for bl in blocks(f))
    return cnt


def dispatch_jobs(num_jobs: int, num_workers: int) -> List[Tuple[int, int]]:
    if num_workers > num_jobs:
        num_workers = num_jobs

    interval = num_jobs // num_workers
    indices = [interval * i for i in range(num_workers+1)]
    indices[-1] = num_jobs
    return [(indices[i], indices[i+1]) for i in range(num_workers)]


def process_line(args: argparse.Namespace, idx_beg: int, idx_end: int):
    intmapping = {}
    if args.map is not None:
        for mapping in args.map:
            if ':' not in mapping:
                raise ValueError(f"No colon ':' found in --map={mapping}")
            index, string = mapping.split(':', maxsplit=1)
            try:
                intmapping[int(index)] = string
            except ValueError:
                raise ValueError(
                    f"failed to read from mapping string \"--mapping={mapping}\"")

    def int2str(x: int) -> str:
        if x in intmapping:
            return intmapping[x]
        else:
            return str(x)

    cachefile = os.path.join('/tmp', str(uuid.uuid4())+'.tmp')
    rm_empty = not args.keep_empty_line
    if args.istext:
        assert args.tokenizer is not None
        tokenizer = tknz.load(args.tokenizer)
        offset = 0
        with open(cachefile, 'w') as fo:
            for file in args.input:
                with open(file, 'r') as fi:
                    for i, line in enumerate(fi):
                        if offset + i < idx_beg:
                            continue
                        if offset + i >= idx_end:
                            break
                        if line.strip() == '' and rm_empty:
                            continue
                        fo.write(' '.join([
                            int2str(x) for x in tokenizer.encode(line)])+'\n')
                offset += countlines(file)
                if offset >= idx_end:
                    break
    else:
        offset = 0
        with open(cachefile, 'w') as fo:
            for file in args.input:
                corpus = CorpusDataset(file)
                for i in range(len(corpus)):
                    if offset + i < idx_beg:
                        continue
                    if offset + i >= idx_end:
                        break
                    fo.write(' '.join([
                        int2str(x) for x in corpus[i][0].tolist()])+'\n')
                offset += len(corpus)
                if offset >= idx_end:
                    break

    return cachefile


def unpack(args):
    return process_line(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs='+',
                        help="Input corpus dataset file.")
    parser.add_argument("-o", "--output", type=str,
                        required=True, help="Output file.")
    parser.add_argument("-t", action="store_true", dest="istext",
                        help="Identify the input to be text instead of binary file. Used with --tokenizer")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model location. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--keep-empty-line", action="store_true",
                        help="Keep empty lines instead removing them (default).")
    parser.add_argument("--map", nargs='*', type=str,
                        help="Map index to str, split by ':'. "
                        "e.g. map 0 to whitespace '--map 0:'; "
                        "     map 0 to whitespace and map 1 to <unk> '--map 0: \"1:<unk>\"'")
    args = parser.parse_args()

    if args.istext:
        total_lines = countlines(args.input)
    else:
        total_lines = sum(len(CorpusDataset(dataset))
                          for dataset in args.input)

    num_process = os.cpu_count()//2
    allargs = dispatch_jobs(total_lines, num_process)
    with Pool(processes=len(allargs)) as pool:
        files_list = pool.map(unpack, [(args,)+d_arg for d_arg in allargs])

    with open(args.output, 'w') as fo:
        for tmpfile in files_list:
            with open(tmpfile, 'r') as fi:
                for bl in blocks(fi):
                    fo.write(bl)
            os.remove(tmpfile)
