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
from ..shared import tokenizer as tknz
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
import pickle
import argparse
from tqdm import tqdm
from typing import Dict, Union, List, Tuple


from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch


def main(args):
    assert os.path.isfile(
        args.nbestlist), f"N-best list file not found: {args.nbestlist}"
    assert args.tokenizer is not None, "You need to specify --tokenizer."
    assert os.path.isfile(
        args.tokenizer), f"Tokenizer model not found: {args.tokenizer}"

    if not torch.cuda.is_available() or args.cpu:
        if args.verbose:
            print("> Using CPU")
        args.cpu = True

    if args.cpu:
        if args.nj == -1:
            world_size = os.cpu_count()
        else:
            world_size = args.nj
    else:
        world_size = torch.cuda.device_count()
    args.world_size = world_size

    cachedir = '/tmp'
    assert os.path.isdir(cachedir), f"Cache directory not found: {cachedir}"
    if not os.access(cachedir, os.W_OK):
        raise PermissionError(f"Permission denied for writing to {cachedir}")
    fmt = os.path.join(cachedir, utils.gen_random_string()+r".{}.tmp")

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

    if args.save_lm_nbest is not None:
        lm_nbest = {}
        with open(args.save_lm_nbest, 'wb') as fo:
            for worker in range(world_size):
                path = fmt.format(worker) + '.lm'
                with open(path, 'rb') as fi:
                    lm_nbest.update(pickle.load(fi))
                os.remove(path)
            pickle.dump(lm_nbest, fo)


def dataserver(args, q: mp.Queue):
    testset = NbestListDataset(args.nbestlist)
    tokenizer = tknz.load(args.tokenizer)
    testloader = DataLoader(
        testset, batch_size=4,
        shuffle=False,
        # FIXME (huahuan): num_workers cannot be set >= 1,
        # possibly pytorch-related bug, see https://github.com/pytorch/pytorch/issues/72782
        num_workers=0,  # (args.world_size // 8),
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
    else:
        device = pid
        torch.cuda.set_device(device)

    if model is None:
        model = build_lm(args.config, args.resume, device)

    writer = fmt.format(pid)
    lm_nbest = {}   # type: Dict[str, Dict[int, Tuple[float, str]]]
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

            if args.save_lm_nbest is not None:
                for _key, _trans, _score in zip(keys, texts, log_lm_probs):
                    nid, okey = _key.split('-', maxsplit=1)
                    if okey not in lm_nbest:
                        lm_nbest[okey] = {}
                    lm_nbest[okey][int(nid)] = (_score, _trans)

            final_score = scores + args.alpha * log_lm_probs.cpu() + args.beta * in_lens
            indiv = {}
            for k, t, s in zip(keys, texts, final_score):
                _, okey = k.split('-', maxsplit=1)
                if okey not in indiv:
                    indiv[okey] = (s, t)
                elif indiv[okey][0] < s:
                    indiv[okey] = (s, t)
            for k, (s, t) in indiv.items():
                fo.write(f"{k} {t}\n")
            del batch

    if args.save_lm_nbest is not None:
        with open(writer+'.lm', 'wb') as fo:
            pickle.dump(lm_nbest, fo)
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

    parser.add_argument("--alpha", type=float, default=0.0,
                        help="The 'alpha' value for LM integration, a.k.a. the LM weight")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="The 'beta' value for LM integration, a.k.a. the penalty of tokens.")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model location. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--save-lm-nbest", type=str,
                        help="Path to save the LM N-best scores.")

    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)

    return parser


if __name__ == "__main__":
    parser = RescoreParser()
    args = parser.parse_args()

    main(args)
