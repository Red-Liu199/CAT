# Author: Huahuan Zheng (maxwellzh@outlook.com)

import sys
import argparse
try:
    import cat
except ModuleNotFoundError:
    import os
    sys.path.append(os.getcwd())
from cat.shared.data import CorpusDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input tokenized text file.")
    args = parser.parse_args()
    libri = CorpusDataset(args.input)
    try:
        for i in range(len(libri)):
            l = [str(x) if x != 1 else ' ' for x in libri[i].tolist()]
            sys.stdout.write(' '.join(l)+'\n')
    except IOError:
        exit(0)
