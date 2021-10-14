'''
Copyright 2021 Tsinghua University, Author: Zheng Huahuan
This script is used for parsing the 'num_classes' in configuration 
according to the sentence piece vocab size.

Usage:
    python parseunits.py sp.vocab config.json
'''
import json
import argparse
import os
import sys


def recursive_rpl(src_dict: dict, target_key: str, rpl_val):
    if not isinstance(src_dict, dict):
        return

    if target_key in src_dict:
        src_dict[target_key] = rpl_val
    else:
        for k, v in src_dict.items():
            recursive_rpl(v, target_key, rpl_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vocab", type=str, help="Vocabulary of SentencePiece model. Each line represents a token.")
    parser.add_argument('config', type=str,
                        help="Configuration file in JSON format.")

    args = parser.parse_args()

    if not os.path.isfile(args.vocab):
        raise RuntimeError(f"Vocabulary file not found: {args.vocab}")

    if not os.path.isfile(args.config):
        raise RuntimeError(f"Configuration file not found: {args.config}")

    with open(args.config, 'r') as fi:
        config = json.load(fi)

    with open(args.vocab, 'r') as fi:
        n_vocab = sum([1 for _ in fi])

    # recursively search for 'num_classes'
    recursive_rpl(config, 'num_classes', n_vocab)

    with open(args.config, 'w') as fo:
        json.dump(config, fo, indent=4)
