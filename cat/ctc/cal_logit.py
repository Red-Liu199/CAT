"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Huahuan Zheng
"""

import kaldiio

from .train import build_model as ctc_builder
from ..shared import coreutils as utils
from ..shared.encoder import AbsEncoder
from ..shared.data import (
    ScpDataset,
    TestPadCollate
)


import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader


def main(args: argparse.Namespace):
    if not os.path.isdir(args.output_dir):
        raise RuntimeError(f"{args.output_dir} is not a directory.")

    if args.nj is None:
        world_size = os.cpu_count()
    else:
        world_size = args.nj
    assert world_size > 0
    args.world_size = world_size

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        print(re)
    q = mp.Queue(maxsize=world_size)
    model = build_model(args)
    model.share_memory()
    mp.spawn(worker, nprocs=world_size+1, args=(args, q, model))

    del q


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


def worker(pid: int, args: argparse.Namespace, q: mp.Queue, model: AbsEncoder):
    if pid == args.world_size:
        return dataserver(args, q)

    torch.set_num_threads(1)

    results = {}
    with torch.no_grad():
        while True:
            batch = q.get(block=True)
            if batch is None:
                break
            key, x, x_lens = batch
            assert len(key) == 1, "Batch size > 1 is not currently support."
            logits, _ = model(x, x_lens)
            log_probs = logits.log_softmax(dim=-1).data.numpy()
            log_probs[log_probs == -np.inf] = -1e16
            results[key[0]] = log_probs[0]
            del batch

    kaldiio.save_ark(os.path.join(
        args.output_dir, f"decode.{pid+1}.ark"), results)
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
    parser.add_argument("--output-dir", type=str, help="Ouput directory.")
    parser.add_argument("--nj", type=int, default=None)
    return parser


if __name__ == '__main__':
    parser = DecoderParser()
    args = parser.parse_args()
    main(args)
