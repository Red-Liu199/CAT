"""Translate the raw text into tokenized IDs.
"""


import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=str,
                        help="Path to the tokenizer file.")
    parser.add_argument("rawtext", type=str, default='', nargs='?',
                        help="Path to the raw text file. (optional)")
    args = parser.parse_args()
    from cat.shared import tokenizer as tknz

    if args.rawtext == '':
        usestd = True
        r_specifier = sys.stdin
    else:
        assert os.path.isfile(args.rawtext), f"{args.rawtext}"
        usestd = False
        r_specifier = open(args.rawtext, 'r')

    try:
        tokenizer = tknz.load(args.tokenizer)
        for line in r_specifier:
            uid, utt = line.strip().split(maxsplit=1)
            sys.stdout.write(
                f"{uid}\t{' '.join(str(x) for x in tokenizer.encode(utt))}\n"
            )
    except IOError:
        pass
    finally:
        if not usestd:
            r_specifier.close()
