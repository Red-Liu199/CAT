# Author: Huahuan Zheng (maxwellzh@outlook.com)

import sys
import argparse
try:
    import cat
except ModuleNotFoundError:
    import os
    sys.path.append(os.getcwd())
from cat.shared.data import CorpusDataset
import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input tokenized text file.")
    parser.add_argument("-t", action="store_true", dest="istext",
                        help="Identify the input to be text instead of binary file. Used with --spm")
    parser.add_argument("--spm", type=str,
                        help="SentencePiece model to tokenize text.")
    args = parser.parse_args()
    if args.istext:
        assert args.spm is not None
        sp = spm.SentencePieceProcessor(model_file=args.spm)
        try:
            for l in sys.stdin:
                idx_l = sp.encode(l)
                l = [str(x) if x != 1 else ' ' for x in sp.encode(l)]
                sys.stdout.write(' '.join(l)+'\n')
        except IOError:
            exit(0)
    else:
        libri = CorpusDataset(args.input)
        try:
            for i in range(len(libri)):
                l = [str(x) if x != 1 else ' ' for x in libri[i].tolist()]
                sys.stdout.write(' '.join(l)+'\n')
        except IOError:
            exit(0)
