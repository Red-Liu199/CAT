"""CTC decode module

Derived from
https://github.com/parlance/ctcdecode
"""

from .train import build_model as ctc_builder
from ..shared import coreutils as utils
from ..shared import tokenizer as tknz
from ..shared.encoder import AbsEncoder
from ..shared.data import (
    ScpDataset,
    TestPadCollate
)
from ctcdecode import CTCBeamDecoder


import os
import json
import time
import pickle
import argparse
from tqdm import tqdm
from typing import Dict, Union, List, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def main(args: argparse.Namespace):

    if args.tokenizer is None or not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(
            "Invalid tokenizer model location: {}".format(args.tokenizer))

    if args.nj == -1:
        world_size = os.cpu_count()
    else:
        world_size = args.nj
    assert world_size > 0
    args.world_size = world_size

    cachedir = '/tmp'
    if not os.access(cachedir, os.W_OK):
        raise PermissionError(f"Permission denied for writing to {cachedir}")
    fmt = os.path.join(cachedir, utils.gen_random_string()+r".{}.tmp")

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        print(re)
    q = mp.Queue(maxsize=world_size)
    model = build_model(args)
    model.share_memory()
    mp.spawn(worker, nprocs=world_size+1, args=(args, q, fmt, model))

    del q
    with open(args.output_prefix, 'w') as fo:
        for i in range(world_size):
            with open(fmt.format(i), 'r') as fi:
                fo.write(fi.read())
            os.remove(fmt.format(i))

    with open(args.output_prefix+'.nbest', 'wb') as fo:
        all_nbest = {}
        for i in range(world_size):
            partial_bin = fmt.format(i) + '.nbest'
            with open(partial_bin, 'rb') as fi:
                partial_nbest = pickle.load(fi)  # type: dict

            all_nbest.update(partial_nbest)
            os.remove(partial_bin)
        pickle.dump(all_nbest, fo)


def dataserver(args, q: mp.Queue):
    testset = ScpDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=args.world_size//8,
        collate_fn=TestPadCollate())

    t_beg = time.time()
    for batch in tqdm(testloader, total=len(testloader)):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for _ in range(args.world_size*2):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print("\nTime = {:.2f} s | RTF = {:.2f} ".format(
        t_dur, t_dur*args.world_size / n_frames * 100))
    time.sleep(2)


def worker(pid: int, args: argparse.Namespace, q: mp.Queue, fmt: str, model: AbsEncoder):
    if pid == args.world_size:
        return dataserver(args, q)
    torch.set_num_threads(args.thread_per_woker)

    tokenizer = tknz.load(args.tokenizer)
    labels = list(tokenizer.dump_vocab().values())

    if args.lm_path is None:
        searcher = CTCBeamDecoder(
            labels, beam_width=args.beam_size, num_processes=1)
    else:
        if args.alpha is None:
            args.alpha = 0.0
        if args.beta is None:
            args.beta = 0.0

        searcher = CTCBeamDecoder(
            labels, model_path=args.lm_path, alpha=args.alpha, beta=args.beta,
            beam_width=args.beam_size, num_processes=1, is_token_based=True)

    local_writer = fmt.format(pid)
    # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
    nbest = {}  # type: Dict[str, Dict[int, Tuple[float, str]]]
    with torch.no_grad(), open(local_writer, 'w') as fi:
        while True:
            batch = q.get(block=True)
            if batch is None:
                break
            key, x, x_lens = batch
            assert len(key) == 1, "Batch size > 1 is not currently support."
            key = key[0]
            probs = torch.softmax(model(x, x_lens)[0], dim=-1)
            beam_results, beam_scores, _, out_lens = searcher.decode(
                probs)
            # make it in decending order
            # -log(p) -> log(p)
            beam_scores = -beam_scores

            nbest[key] = {
                bid: (score, tokenizer.decode(hypo[:_len].tolist()))
                for bid, (score, hypo, _len) in enumerate(zip(beam_scores[0], beam_results[0], out_lens[0]))
            }
            fi.write("{} {}\n".format(key, nbest[key][0][1]))
            del batch

    with open(f"{local_writer}.nbest", 'wb') as fo:
        pickle.dump(nbest, fo)
    q.get()


def build_model(args: argparse.Namespace):
    assert args.resume is not None, "Trying to decode with uninitialized parameters. Add --resume"

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = ctc_builder(None, configures, dist=False)
    model = utils.load_checkpoint(model, args.resume)
    model = model.am
    model.eval()
    return model


def DecoderParser():
    parser = utils.BasicDDPParser(
        istraining=False, prog='CTC decoder.')

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default='./decode')
    parser.add_argument("--lm-path", type=str, help="Path to KenLM model.")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="The 'alpha' value for LM integration, a.k.a. the LM weight")
    parser.add_argument("--beta", type=float, default=0.6,
                        help="The 'beta' value for LM integration, a.k.a. the penalty of tokens.")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model location. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    return parser


if __name__ == '__main__':
    parser = DecoderParser()
    args = parser.parse_args()
    main(args)
