""" Compute ppl on specified test sets of n-gram model.

TODO:
    deprecated this function and make it the same as NN-LM pipeline
"""
import os
import json
import argparse

TESTBIN = "lmbin/test.pkl"


def main(args: argparse.Namespace):
    try:
        import cat
    except ModuleNotFoundError:
        import sys
        sys.path.append(os.getcwd())
        from cat.shared.data import CorpusDataset
        from cat.lm import lm_builder

    with open(f'{args.dir}/config.json', 'r') as fi:
        configures = json.load(fi)
    model = lm_builder(None, configures, dist=False, wrapper=False)
    model.eval()

    for testset in args.apd:
        testdata = CorpusDataset(f"{testset}/{TESTBIN}")

        log_probs = 0.
        n_tokens = 0
        for i in range(len(testdata)):
            inputs, targets = testdata[i]
            scores = model.score(inputs.unsqueeze(0), targets.unsqueeze(
                0), inputs.new_tensor([inputs.size(0)]))
            log_probs += scores
            n_tokens += inputs.size(0)
            print(f"\r[{i+1}/{len(testdata)}]", end='')
        print("")

        ppl = (-log_probs/n_tokens).exp()
        print("PPL over {} test set is: {:.2f}".format(
            testset.split('/')[1], ppl.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Location of directory.")
    parser.add_argument("--apd", type=str, nargs='*',
                        help="Experiment directories of extra test sets.")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"{args.dir} does not exist!")

    if args.apd is None:
        args.apd = [args.dir]
    else:
        args.apd = list(set([args.dir] + args.apd))

    for _set in args.apd:
        _path = os.path.join(_set, TESTBIN)
        if not os.path.isfile(_path):
            raise FileNotFoundError(f"{_path} does not exist!")

    main(args)
