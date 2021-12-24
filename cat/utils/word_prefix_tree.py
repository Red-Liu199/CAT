"""Construct word prefix tree from text corpus.

    Author: Huahuan Zheng (maxwellzh@outlook.com)
"""

import sentencepiece as spm
import argparse
import sys
import os
from typing import List, Tuple


def main(args: argparse.Namespace = None):
    try:
        import cat
    except ModuleNotFoundError:
        sys.path.append(os.getcwd())
    from cat.rnnt.beam_search_transducer import PrefixTree

    f_raw_text = args.intext
    f_spmodel = args.spmodel

    spmodel = spm.SentencePieceProcessor(model_file=f_spmodel)

    with open(f_raw_text, 'r') as fi:
        seq_lines = fi.readlines()

    # rm seq id
    if args.stripid:
        seq_lines = [l.split()[1:]
                     for l in seq_lines]      # type: List[List[str]]
    else:
        seq_lines = [l.split() for l in seq_lines]

    words = []  # type: List[List[str]]
    for l in seq_lines:
        words += l

    print(f"# words: {len(words)}")
    words = list(set(words))
    print(f"# unique words: {len(words)}")

    idx_words = []  # type: List[Tuple[List[int], str]]
    for w in words:
        tokenized_w = spmodel.encode(w)
        idx_words.append((tokenized_w, w))

    prefix_tree = PrefixTree()
    for idx, w in idx_words:
        if idx not in prefix_tree:
            prefix_tree.update(idx, w)

    print(f"# element in prefix tree: {prefix_tree.size()}")

    print(f"Mem of text words: {sys.getsizeof(words)} Byte")
    print(
        f"Mem of tokenized words: {sys.getsizeof(idx_words)-sys.getsizeof(words)} Byte")
    print(f"Mem of treed words: {sys.getsizeof(prefix_tree)} Byte")

    if args.output is not None:
        prefix_tree.save(args.output)
        print(f"save word prefix tree binary at '{args.output}'")


def WordPrefixParser():
    parser = argparse.ArgumentParser("Word prefix tree constructor.")
    parser.add_argument("intext", type=str,
                        help="Input text. Each line represent a sentence.")
    parser.add_argument("spmodel", type=str,
                        help="Path to the SentencePiece model.")
    parser.add_argument("--stripid", action="store_true",
                        help="Flag to specify whether strip the first token at the beginning of each line.")
    parser.add_argument("--output", type=str,
                        help="Dump the prefix tree to file, optional.")
    return parser


if __name__ == "__main__":
    parser = WordPrefixParser()
    args = parser.parse_args()
    main(args)
