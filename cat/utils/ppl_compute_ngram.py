""" Compute ppl on specified test sets of n-gram model.

TODO:
    deprecated this function and make it the same as NN-LM pipeline
"""
import os
import json
import math
import uuid
import shutil
import argparse
from multiprocessing import Pool
from typing import Tuple
try:
    import cat
except ModuleNotFoundError:
    import sys
    sys.path.append(os.getcwd())

TESTBIN = "lmbin/test.pkl"


def evaluate(model, dataset: str, idx_beg: int, idx_end: int) -> Tuple[float, int]:
    """Evaluate dataset in [idx_beg, idx_end) range, and return sum of logprobs and tokens."""

    from cat.shared.data import CorpusDataset
    testdata = CorpusDataset(dataset)

    log_probs = 0.
    n_tokens = 0
    for i in range(idx_beg, idx_end):
        inputs, targets = testdata[i]
        scores = model.score(inputs.unsqueeze(0), targets.unsqueeze(0))
        log_probs += scores
        n_tokens += inputs.size(0)

    return (log_probs.item(), n_tokens)


def unpack_args(args):
    return evaluate(*args)


def text2corpusbin(f_text: str, f_bin: str, tokenizer):
    from asr_process import updateNamespaceFromDict
    from transText2Bin import main as t2bmain
    from transText2Bin import TextProcessingParser as t2bparser

    t2bmain(
        updateNamespaceFromDict(
            {
                'tokenizer': tokenizer,
                'quiet': True
            }, t2bparser(), [f_text, f_bin]))

    return


def main(args: argparse.Namespace):
    if args.tokenizer is not None:
        assert os.path.isfile(
            args.tokenizer), f"no such tokenizer file: '{args.tokenizer}'"
        cachedir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(cachedir)
    else:
        cachedir = None

    from cat.shared.data import CorpusDataset
    from cat.lm import lm_builder

    with open(f'{args.dir}/config.json', 'r') as fi:
        configures = json.load(fi)
    model = lm_builder(None, configures, dist=False, wrapper=False)
    model.eval()
    model.share_memory()
    # multi-processing compute
    num_threads = 40  # int(os.cpu_count())

    for testset in args.evaluate:
        if args.tokenizer is not None:
            binfile = os.path.join(cachedir, f"{str(uuid.uuid4())}.pkl.tmp")
            text2corpusbin(testset, binfile, args.tokenizer)
        else:
            binfile = testset

        num_lines = len(CorpusDataset(binfile))
        interval = num_lines // num_threads
        indices = [interval * i for i in range(num_threads+1)]
        if indices[-1] != num_lines:
            indices[-1] = num_lines
        pool_args = [(model, binfile, indices[i], indices[i+1])
                     for i in range(num_threads)]
        with Pool(processes=num_threads) as pool:
            gather_output = pool.map(unpack_args, pool_args)

        log_probs, num_tokens = list(zip(*gather_output))
        ppl = math.exp(-sum(log_probs)/sum(num_tokens))
        print("Test set: {} -> ppl: {:.2f}".format(testset, ppl))

    if cachedir is not None:
        shutil.rmtree(cachedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Location of directory.")
    parser.add_argument("-e", "--evaluate", type=str, nargs='*',
                        help="Evaluate test sets.")
    parser.add_argument("--tokenizer", type=str,
                        help="Use tokenizer to encode the evaluation sets. If passed, would take -e inputs as text files.")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"{args.dir} does not exist!")

    if args.evaluate is None:
        args.evaluate = [os.path.join(args.dir, TESTBIN)]

    for _path in args.evaluate:
        if not os.path.isfile(_path):
            raise FileNotFoundError(f"{_path} does not exist!")

    main(args)
