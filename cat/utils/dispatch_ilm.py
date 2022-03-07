"""
Dispatch the N-best list with ILM scores into two:
1. RNN-T scores only, would replace the original file by default.
2. ILM scores, named as ${file}.ilm
"""
import pickle
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str,
                        help="Input N-best list file with ILM scores.")
    parser.add_argument("--save-location", type=str, help="specify the output dispatched N-best file location, "
                        "the ILM Nbest list file would be ${save-location}.ilm, default: ${input}")
    args = parser.parse_args()

    src_nbest = args.input
    with open(src_nbest, 'rb') as fi:
        data = pickle.load(fi)

    am_score = {}
    ilm_score = {}
    for key, hypos in data.items():
        if key[-4:] == '-ilm':
            ilm_score[key[:-4]] = hypos
        else:
            am_score[key] = hypos

    if args.save_location is None:
        dst_nbest = src_nbest
    else:
        dst_nbest = args.save_location

    with open(dst_nbest, 'wb') as fo:
        pickle.dump(am_score, fo)

    with open(dst_nbest+'.ilm', 'wb') as fo:
        pickle.dump(ilm_score, fo)
