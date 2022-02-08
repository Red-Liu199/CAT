# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Rescoring with custom LM.

This script is ported from cat.rnnt.decode

Rescore N-best list.

P.S. CPU is faster when rescoring with n-gram model, while GPU
     would be faster rescoring with NN model.
"""

from ..shared import coreutils as utils
from . import lm_builder
from ..shared.decoder import (
    AbsDecoder,
    NGram
)
from ..shared.data import (
    NbestListDataset,
    NbestListCollate
)

import os
import json
import time
import uuid
import argparse
import sentencepiece as sp
from tqdm import tqdm


from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch


def main(args):
    assert os.path.isfile(
        args.nbestlist), f"N-best list file not found: {args.nbestlist}"
    assert args.spmodel is not None, "You need to specify --spmodel."
    assert os.path.isfile(
        args.spmodel), f"SentencePiece model not found: {args.spmodel}"

    if not torch.cuda.is_available() or args.cpu:
        if args.verbose:
            print("> Using CPU")
        args.cpu = True

    if args.cpu:
        if args.nj is None:
            world_size = os.cpu_count()
        else:
            world_size = args.nj
    else:
        world_size = torch.cuda.device_count()
    args.world_size = world_size

    randomstr = str(uuid.uuid4())
    cachedir = '/tmp'
    assert os.path.isdir(cachedir), f"Cache directory not found: {cachedir}"
    fmt = os.path.join(cachedir, randomstr+r".{}.tmp")

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        if args.verbose:
            print(re)
    q = mp.Queue(maxsize=world_size)

    if args.cpu:
        model = build_lm(args.config, args.resume, 'cpu')
        model.share_memory()
        mp.spawn(main_worker, nprocs=world_size+1,
                 args=(args, q, fmt, model))
    else:
        mp.spawn(main_worker, nprocs=world_size+1,
                 args=(args, q, fmt))
    del q

    with open(args.output, 'w') as fo:
        for worker in range(world_size):
            path = fmt.format(worker)
            with open(path, 'r') as fi:
                fo.write(fi.read())
            os.remove(path)


def dataserver(args, q: mp.Queue):
    testset = NbestListDataset(args.nbestlist)

    if args.tokenizer == 'sentencepiece':
        tokenizer = sp.SentencePieceProcessor(model_file=args.spmodel)
    elif args.tokenizer == 'jieba':
        from ..shared.tokenizer import JiebaTokenizer
        tokenizer = JiebaTokenizer(userdict=args.spmodel)
    else:
        raise RuntimeError(
            f"Unknown tokenizer type \'{args.tokenizer}\', expected one of ['sentencepiece', 'jieba']")

    testloader = DataLoader(
        testset, batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=NbestListCollate(tokenizer))

    t_beg = time.time()
    for batch in tqdm(testloader, total=len(testloader), disable=(not args.verbose)):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for i in range(args.world_size*2):
        q.put(None, block=True)
    t_end = time.time()
    if args.verbose:
        print("\nTime of rescoring: {:.2f}s".format(t_end-t_beg))
    time.sleep(2)


def main_worker(pid: int, args: argparse.Namespace, q: mp.Queue, fmt: str = "rescore.{}.tmp", model=None):
    if pid == args.world_size:
        return dataserver(args, q)
    args.pid = pid
    args.rank = pid
    world_size = args.world_size

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(1)
        dist.init_process_group(
            backend='gloo', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)
    else:
        device = pid
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)

    if model is None:
        model = build_lm(args.config, args.resume, device)

    writer = fmt.format(pid)
    # rescoring
    with torch.no_grad(), \
            autocast(enabled=(True if device != 'cpu' else False)), open(writer, 'w') as fo:
        while True:
            batch = q.get(block=True)
            if batch is None:
                break
            keys, texts, scores, in_toks, mask = batch
            in_toks = in_toks.to(device)

            # suppose </s> = <s>
            dummy_targets = torch.roll(in_toks, -1, dims=1)
            in_lens = in_toks.size(1) - mask.sum(dim=1)
            log_lm_probs = model.score(in_toks, dummy_targets, in_lens)

            final_score = scores + args.alpha * log_lm_probs.cpu() + args.beta * in_lens
            indiv = {}
            for k, t, s in zip(keys, texts, final_score):
                if k not in indiv:
                    indiv[k] = (s, t)
                elif indiv[k][0] < s:
                    indiv[k] = (s, t)
            for k, (s, t) in indiv.items():
                fo.write(f"{k} {t}\n")
            del batch

    q.get()


def build_lm(f_config: str, f_check: str, device='cuda') -> AbsDecoder:
    if isinstance(device, int):
        device = f'cuda:{device}'

    with open(f_config, 'r') as fi:
        configures = json.load(fi)
    model = lm_builder(None, configures, dist=False)
    if not isinstance(model.lm, NGram):
        model = utils.load_checkpoint(model.to(device), f_check)
    model = model.lm
    model.eval()
    return model


def RescoreParser():
    parser = utils.BasicDDPParser(
        istraining=False, prog="Rescore with give n-best list and LM")

    parser.add_argument("nbestlist", type=str,
                        help="Path to N-best list files.")
    parser.add_argument("output", type=str, help="The output text file. ")

    parser.add_argument("--alpha", type=float, default=0.3,
                        help="The 'alpha' value for LM integration, a.k.a. the LM weight")
    parser.add_argument("--beta", type=float, default=0.6,
                        help="The 'beta' value for LM integration, a.k.a. the penalty of tokens.")
    # TODO (huahun): rename this option to tokenizer-file
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")
    parser.add_argument("--tokenizer", type=str, choices=['sentencepiece', 'jieba'], default='sentencepiece',
                        help="Specify which tokenizer to use.")

    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)

    return parser


if __name__ == "__main__":
    parser = RescoreParser()
    args = parser.parse_args()

    main(args)
