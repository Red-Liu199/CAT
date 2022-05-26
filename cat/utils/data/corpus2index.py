# Author: Huahuan Zheng (maxwellzh@outlook.com)


import io
import os
import sys
import argparse
from typing import *
from multiprocessing import Process, Queue


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


class TextLoader:
    def __init__(self, f_tknz: str) -> None:
        self._tknz = tknz.load(f_tknz)

    def __call__(self, corpus: str, _offset: int = 0, _cnt: int = -1):
        if _offset >= countlines(corpus):
            return
        with open(corpus, 'r') as fi:
            for i, line in enumerate(fi):
                if i < _offset:
                    continue
                if i == _cnt:
                    break
                indices = self._tknz.encode(line.strip())
                yield self._tknz.encode(line.strip())
        return


class BinLoader:
    def __call__(self, corpus: str, _offset: int = 0, _cnt: int = -1):
        data = CorpusDataset(corpus)
        if _offset >= len(data):
            return
        for i in range(len(data)):
            if i < _offset:
                continue
            if i == _cnt:
                break
            yield data[i][0].tolist()
        return


def process_worker(args: argparse.Namespace, p_range: Tuple[int, int], q_out: Queue, mapping: Dict[int, str] = {}):
    def _int2str(x: int) -> str:
        return mapping.get(x, str(x))

    idx_beg, idx_end = p_range
    rm_empty = not args.keep_empty_line
    if args.istext:
        loader = TextLoader(args.tokenizer)
    else:
        loader = BinLoader()

    offset = 0
    buffer = io.StringIO()
    for file in args.input:
        for tokens in loader(file, idx_beg - offset, idx_end - offset):
            if tokens == [] and rm_empty:
                continue
            buffer.write(' '.join(_int2str(x) for x in tokens)+'\n')
            if buffer.tell() >= 16777216:
                q_out.put(buffer.getvalue(), block=True)
                buffer.truncate(0)
        offset += countlines(file)
        if offset >= idx_end:
            break

    q_out.put(buffer.getvalue(), block=True)
    buffer.close()
    q_out.put(None, block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs='+',
                        help="Input corpus dataset file.")
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

    for _in in args.input:
        assert _in != '/dev/stdin', f"redirect /dev/stdin would cause unexpected behavior."

    mapping = {}
    if args.map is not None:
        for _m in args.map:
            if ':' not in _m:
                raise ValueError(f"No colon ':' found in --map={_m}")
            index, string = _m.split(':', maxsplit=1)
            try:
                mapping[int(index)] = string
            except ValueError:
                raise ValueError(
                    f"failed to read from mapping string \"--mapping={mapping}\"")

    from cat.shared.data import CorpusDataset
    from cat.shared import tokenizer as tknz
    if args.istext:
        total_lines = countlines(args.input)
    else:
        total_lines = sum(len(CorpusDataset(dataset))
                          for dataset in args.input)

    num_process = max(min(os.cpu_count()//2, total_lines//10000), 1)
    workerloads = dispatch_jobs(total_lines, num_process)

    try:
        q = Queue(maxsize=1)
        p = []
        for i in range(num_process):
            p.append(Process(
                target=process_worker,
                args=(args, workerloads[i], q, mapping)
            ))
            p[-1].start()

        cnt_done = 0
        while True:
            line = q.get(block=True)
            if line is None:
                cnt_done += 1
                if cnt_done == num_process:
                    break
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
            del line

        for _p in p:
            _p.join()

    except IOError:
        pass
    finally:
        for _p in p:
            _p.terminate()
        del q
