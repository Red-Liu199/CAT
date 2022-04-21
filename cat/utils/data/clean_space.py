"""
Author: Huahuan Zheng

This is a script used for cleaning the spaces of text corpus (useful for Asian-lang)
"""

import argparse
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("intext", type=str, nargs='+', help="Text corpus")
    parser.add_argument("-o", "--output", type=str, nargs='*', default=None,
                        help="Specify the output file(s), "
                        "if specified, should be the same number of inputs. "
                        "If not specified, input files are modified in-place.")
    parser.add_argument("-i", "--skip-id", action="store_true",
                        help="Skip the first column as utterance id.")

    args = parser.parse_args()

    for f in args.intext:
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"One of input: '{f}' is not a valid file.")

    if args.output is None:
        inplace = True
        # create temporary files for changing the contents
        cachedir = os.path.join('/tmp', 'clean-space')
        os.makedirs(cachedir, exist_ok=True)
        f_out = [os.path.join(cachedir, os.path.basename(f))
                 for f in args.intext]
    else:
        if len(args.output) != len(args.intext):
            raise ValueError(
                f"If you specify -o/--ouput, the number of -o args should match that of input, insead {len(args.output)} != {len(args.intext)} ")
        inplace = False
        f_out = args.output
        cachedir = None

    for f_i, f_o in zip(args.intext, f_out):
        with open(f_i, 'r') as fit, open(f_o, 'w') as fot:
            if args.skip_id:
                for line in fit:
                    uid, utt = line.split(maxsplit=1)
                    fot.write(f"{uid}\t{utt.replace(' ', '')}")
                    pass
            else:
                for line in fit:
                    fot.write(line.replace(' ', ''))

    if inplace:
        for f_i, f_o in zip(args.intext, f_out):
            shutil.move(f_o, f_i)
        shutil.rmtree(cachedir)
