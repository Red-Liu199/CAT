# Author: Huahuan Zheng (maxwellzh@outlook.com)
#
# Prepare the extra corpus from wenetspeech train-l
#
import sys
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str,
                        help="Path to the wenetspeech train-L text corpus.")
    parser.add_argument("--exclude-corpus", type=str, dest="f_exc",
                        help="Specify if you want to exclude part of the corpus, take first column as index.")
    parser.add_argument("-n", "--num-lines", type=int,
                        help="Number of lines to be prepared, if not specified, would take all of them.")
    args = parser.parse_args()

    if not os.path.isfile(args.corpus):
        raise FileNotFoundError(f"--corpus={args.corpus} is not a valid file.")

    if args.f_exc is not None and not os.path.isfile(args.f_exc):
        raise FileNotFoundError(
            f"--exclude-corpus={args.f_exc} is not a valid file.")

    if args.num_lines is not None:
        if args.num_lines < 0:
            raise ValueError(
                f"--num-lines={args.num_lines} < 0 is invalid, expected valud >= 0")
        num_lines = args.num_lines
    else:
        num_lines = sum(1 for _ in open(args.corpus, 'r'))

    # prepare excluding list
    excluding_list = set()
    if args.f_exc is not None:
        with open(args.f_exc, 'r') as fi:
            for line in fi:
                line = line.strip()
                if ' ' in line or '\t' in line:
                    uid, _ = line.split(maxsplit=1)
                else:
                    uid = line
                excluding_list.add(uid)

    cnt = 0
    with open(args.corpus, 'r') as fi:
        try:
            for line in fi:
                line = line.strip()
                if ' ' in line or '\t' in line:
                    uid, _ = line.split(maxsplit=1)
                else:
                    uid = line
                if uid in excluding_list:
                    continue

                if cnt >= num_lines:
                    break
                sys.stdout.write(f"{line}\n")
                cnt += 1
        except IOError:
            exit(0)

    if cnt < num_lines:
        raise RuntimeError(
            f"Source corpus text doesn't have enough unique lines to export: {cnt} in total, expect {num_lines}")
