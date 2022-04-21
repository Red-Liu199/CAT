"""Construct word prefix tree from text corpus.

    Author: Huahuan Zheng (maxwellzh@outlook.com)
"""

from cat.rnnt.beam_search_transducer import PrefixTree
from cat.shared import tokenizer as tknz

import argparse
import sys
import os
from typing import List, Tuple


def main(args: argparse.Namespace = None):

    f_raw_text = args.intext
    tokenizer = tknz.load(args.tokenizer)

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
        tokenized_w = tokenizer.encode(w)
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
    parser.add_argument("tokenizer", type=str,
                        help="Tokenizer model location. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--stripid", action="store_true",
                        help="Flag to specify whether strip the first token at the beginning of each line.")
    parser.add_argument("--output", type=str,
                        help="Dump the prefix tree to file, optional.")
    return parser


if __name__ == "__main__":
    parser = WordPrefixParser()
    args = parser.parse_args()
    main(args)
