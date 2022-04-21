""" Compute ppl on specified test sets with LM.

"""

from ..shared.data import (
    CorpusDataset,
    sortedPadCollateLM
)
from ..shared import coreutils
from . import lm_builder

import os
import sys
import math
import uuid
import shutil
import argparse
from multiprocessing import Pool
from typing import Tuple, List

import gather

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def main(args: argparse.Namespace):
    for _path in args.evaluate:
        if not os.path.isfile(_path):
            raise FileNotFoundError(f"{_path} does not exist!")

    if args.tokenizer is not None:
        assert os.path.isfile(
            args.tokenizer), f"no such tokenizer file: '{args.tokenizer}'"
        assert os.access('/tmp', os.W_OK), f"/tmp is non-writable."
        cachedir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(cachedir)
    else:
        cachedir = None

    isngram = isNGram(args)
    if (not isngram) and torch.cuda.is_available():
        usegpu = True
    else:
        usegpu = False

    if args.nj == -1:
        if usegpu:
            world_size = torch.cuda.device_count()
        else:
            world_size = os.cpu_count() // 2
    else:
        world_size = args.nj
    assert world_size > 0
    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        print(re)
    q = mp.Queue(maxsize=world_size)

    args.usegpu = usegpu
    processed_files = []
    for testset in args.evaluate:
        if args.tokenizer is not None:
            binfile = os.path.join(cachedir, f"{str(uuid.uuid4())}.pkl.tmp")
            text2corpusbin(testset, binfile, args.tokenizer)
        else:
            binfile = testset
        processed_files.append(binfile)

    if usegpu:
        mp.spawn(evaluate_nnlm, nprocs=(world_size+1),
                 args=(world_size, q, args, processed_files))
    else:
        model = build_model(args, 'cpu')
        model.share_memory()
        if isngram:
            mp.spawn(evaluate_ngram, nprocs=(world_size+1),
                     args=(world_size, q, args, processed_files, model))
        else:
            mp.spawn(evaluate_nnlm, nprocs=(world_size+1),
                     args=(world_size, q, args, processed_files, model))

    if cachedir is not None:
        shutil.rmtree(cachedir)


@torch.no_grad()
def evaluate_ngram(pid: int, wsize: int, q: mp.Queue, args: argparse.Namespace, testsets: List[str], model):
    """Evaluate datasets and return sum of logprobs and tokens."""
    if pid == wsize:
        return consumer(wsize, q, args)

    torch.set_num_threads(1)
    output = []     # type: List[Tuple[float, int]]
    for f_data in testsets:
        testdata = CorpusDataset(f_data)
        log_probs = 0.
        n_tokens = 0
        for i in range(pid * (len(testdata) // wsize), (pid+1) * (len(testdata) // wsize)):
            inputs, targets = testdata[i]
            scores = model.score(inputs.unsqueeze(0), targets.unsqueeze(0))
            log_probs += scores
            n_tokens += inputs.size(0)
        output.append((log_probs.item(), n_tokens))
    q.put(output)


@torch.no_grad()
def evaluate_nnlm(pid: int, wsize: int, q: mp.Queue, args: argparse.Namespace, testsets: List[str], model=None):
    """Evaluate datasets and return sum of logprobs and tokens."""
    if pid == wsize:
        return consumer(wsize, q, args)

    if args.usegpu:
        device = pid
        torch.cuda.set_device(device)
        model = build_model(args, device)
    else:
        torch.set_num_threads(1)
        device = 'cpu'
        assert model is not None
    assert next(iter(model.parameters())).device == torch.device(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    output = []     # type: List[Tuple[float, int]]
    for f_data in testsets:
        testdata = CorpusDataset(f_data)
        # slice the dataset to avoid duplicated datas
        testdata.offsets = testdata.offsets[
            pid * (len(testdata)//wsize):(pid+1)*(len(testdata)//wsize)]
        testloader = DataLoader(
            testdata, batch_size=32, collate_fn=sortedPadCollateLM())
        nll = 0.
        n_tokens = 0
        for minibatch in testloader:
            features, input_lengths, labels, _ = minibatch
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds, _ = model(features, input_lengths=input_lengths)
            # gather op doesn't support cpu
            if device == 'cpu':
                logits = torch.cat([
                    preds[n, :input_lengths[n], :]
                    for n in range(preds.size(0))
                ], dim=0)
            else:
                logits = gather.cat(preds, input_lengths)
            nll += criterion(logits, labels) * logits.size(0)
            n_tokens += logits.size(0)
        output.append((-nll.item(), n_tokens))

    q.put(output, block=True)
    # time.sleep(10)


def consumer(wsize: int, q: mp.Queue, args):
    output = []     # type: List[List[Tuple[float, int]]]
    for i in range(wsize):
        data = q.get(block=True)
        output.append(data)

    for i_f, f_data in enumerate(args.evaluate):
        ppl = math.exp(
            -sum(
                output[i_worker][i_f][0]
                for i_worker in range(wsize)
            )/sum(
                output[i_worker][i_f][1]
                for i_worker in range(wsize)
            )
        )
        sys.stdout.write("Test file: {} -> ppl: {:.2f}\n".format(f_data, ppl))


def text2corpusbin(f_text: str, f_bin: str, tokenizer):
    from ..utils.pipeline.asr import updateNamespaceFromDict
    from ..utils.data import transText2Bin as t2b
    t2b.main(
        updateNamespaceFromDict(
            {
                'tokenizer': tokenizer,
                'quiet': True
            }, t2b.TextProcessingParser(), [f_text, f_bin]))

    return


def isNGram(args):
    configures = coreutils.readjson(args.config)
    return configures['decoder']['type'] == 'NGram'


def build_model(args: argparse.Namespace, device):
    configures = coreutils.readjson(args.config)
    isngram = (configures['decoder']['type'] == 'NGram')
    if not isngram:
        model = lm_builder(configures, dist=False, wrapper=True)
        if args.resume is None:
            sys.stderr.write(
                f"You're trying to compute ppl with un-initialized model.\n"
                f"... ensure you know what you're doing.\n")
        else:
            coreutils.load_checkpoint(model, args.resume)
        # squeeze the wrapper
        model = model.lm
    else:
        model = lm_builder(configures, dist=False, wrapper=False)

    model.eval()
    model = model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str,
                        help="Path to the configuration file, usually 'path/to/config.json")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("-e", "--evaluate", type=str, nargs='+', required=True,
                        help="Evaluate test sets. w/o --tokenizer, -e inputs are assumed to be CorpusDataset format binary data.")
    parser.add_argument("--tokenizer", type=str,
                        help="Use tokenizer to encode the evaluation sets. If passed, would take -e inputs as text files.")
    parser.add_argument("--resume", type=str,
                        help="Path to the checkpoint of NNLM, not required for N-gram LM.")
    args = parser.parse_args()

    main(args)
