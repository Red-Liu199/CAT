# Author: Huahuan Zheng (maxwellzh@outlook.com)

import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str,
                        help="Input corpus dataset file. Ignored if use -t")
    parser.add_argument("-t", action="store_true", dest="istext",
                        help="Identify the input to be text instead of binary file. Read from stdin, used with --tokenizer")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model location. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--map", nargs='*', type=str,
                        help="Map index to str, split by ':'. "
                        "e.g. map 0 to whitespace '--map 0:'; "
                        "     map 0 to whitespace and map 1 to <unk> '--map 0: \"1:<unk>\"'")
    args = parser.parse_args()
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

    try:
        import cat
    except ModuleNotFoundError:
        import os
        sys.path.append(os.getcwd())
    if args.istext:
        from cat.shared import tokenizer as tknz
        assert args.tokenizer is not None
        tokenizer = tknz.load(args.tokenizer)
        try:
            for l in sys.stdin:
                idx_l = tokenizer.encode(l)
                sys.stdout.write(' '.join([
                    int2str(x) for x in tokenizer.encode(l)
                ]) + '\n')
        except IOError:
            exit(0)
    else:
        from cat.shared.data import CorpusDataset
        corpus = CorpusDataset(args.input)
        try:
            for i in range(len(corpus)):
                sys.stdout.write(' '.join([
                    int2str(x) for x in corpus[i][0].tolist()
                ])+'\n')
        except IOError:
            exit(0)
