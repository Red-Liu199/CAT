# Author: Huahuan Zheng (maxwellzh@outlook.com)

import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input tokenized text file.")
    parser.add_argument("-t", action="store_true", dest="istext",
                        help="Identify the input to be text instead of binary file. Used with --spm")
    parser.add_argument("--spm", type=str,
                        help="SentencePiece model to tokenize text.")
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

    if args.istext:
        import sentencepiece as spm
        assert args.spm is not None
        sp = spm.SentencePieceProcessor(model_file=args.spm)
        try:
            for l in sys.stdin:
                idx_l = sp.encode(l)
                sys.stdout.write(' '.join([
                    int2str(x) for x in sp.encode(l)
                ]) + '\n')
        except IOError:
            exit(0)
    else:
        try:
            import cat
        except ModuleNotFoundError:
            import os
            sys.path.append(os.getcwd())
        from cat.shared.data import CorpusDataset
        corpus = CorpusDataset(args.input)
        try:
            for i in range(len(corpus)):
                sys.stdout.write(' '.join([
                    int2str(x) for x in corpus[i][0].tolist()
                ])+'\n')
        except IOError:
            exit(0)
