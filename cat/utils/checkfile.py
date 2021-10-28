'''
Author: Zheng Huahuan
This script is used to check whether files/directories exist.
Usage;
    python checkfile.py -d A/ B/ C/ -f a.a b.b c.c
'''
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default=[], nargs='+',
                        dest='dirs', help="Directories.")
    parser.add_argument('-f', type=str, default=[],
                        nargs='+', dest='files', help="Files")

    args = parser.parse_args()

    not_founds = {'d': [], 'f': []}
    for d in args.dirs:
        if not os.path.isdir(d):
            not_founds['d'].append(d)

    for f in args.files:
        if not os.path.isfile(f):
            not_founds['f'].append(f)

    if len(not_founds['d']) > 0:
        print("Directory checking failed:")
        for d in not_founds['d']:
            print(f"    {d}")

    if len(not_founds['f']) > 0:
        print("Files checking failed:")
        for f in not_founds['f']:
            print(f"    {f}")

    if len(not_founds['d']) == 0 and len(not_founds['f']) == 0:
        exit(0)
    else:
        exit(1)
