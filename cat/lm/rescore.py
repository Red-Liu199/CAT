# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Rescoring with custom LM.

This script is ported from cat.rnnt.decode

Rescore N-best list
"""

from ..shared import coreutils as utils
from . import lm_builder
from ..shared.decoder import AbsDecoder
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
from typing import Optional


from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch


class ModelConfig:
    def __init__(self, f_config: Optional[str] = None, f_check: Optional[str] = None) -> None:
        if f_config is not None:
            assert os.path.isfile(f_config)
        self.f_config = f_config
        if f_check is not None:
            assert os.path.isfile(f_check)
        self.f_check = f_check


def main(args):
    assert os.path.isfile(args.nbestlist), f"File not found: {args.nbestlist}"
    assert args.lamb >= 0.0, f"Consider using --ILM instead of negative --lamb={args.lamb}"
    assert args.spmodel is not None, "You need to specify --spmodel."
    assert os.path.isfile(args.spmodel), f"File not found: {args.spmodel}"

    if not torch.cuda.is_available() or args.cpu:
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
        print(re)
    q = mp.Queue(maxsize=world_size*2)

    if args.cpu:
        model = build_lm(ModelConfig(args.lm_config, args.lm_check), 'cpu')
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
    tokenizer = sp.SentencePieceProcessor(model_file=args.spmodel)
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=args.world_size//8,
        collate_fn=NbestListCollate(tokenizer))

    t_beg = time.time()
    for batch in tqdm(testloader, total=len(testloader)):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for i in range(args.world_size*2):
        q.put(None, block=True)
    t_end = time.time()
    print("\nTime of searching: {:.2f}s".format(t_end-t_beg))
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
        model = build_lm(ModelConfig(args.lm_config, args.lm_check), device)

    writer = fmt.format(pid)
    # rescoring
    with autocast(enabled=(True if device != 'cpu' else False)), open(writer, 'w') as fo:
        while True:
            batch = q.get(block=True)
            if batch is None:
                break
            keys, texts, scores, input, mask = batch
            input, mask = input.to(device), mask.to(device)

            logits, _ = model(input)
            log_p = logits.log_softmax(dim=-1)
            ll = log_p.gather(dim=-1, index=input.unsqueeze(-1)
                              ).squeeze(-1)    # type:torch.Tensor
            ll = ll.masked_fill_(mask, float(0.0))
            log_p_lm = args.lamb * ll.sum(dim=-1)

            # length norm
            l_y = input.size(1) - mask.sum(dim=-1)
            log_p_lm += 0.6 * l_y

            rescored_list = {}
            for k, t, ac_score, lm_score in zip(keys, texts, scores, log_p_lm.cpu()):
                new_score = ac_score + lm_score
                if k not in rescored_list:
                    rescored_list[k] = (new_score, t)
                else:
                    if new_score > rescored_list[k][0]:
                        rescored_list[k] = (new_score, t)

            for key, (score, seq) in rescored_list.items():
                fo.write(f"{key} {seq}\n")
            del batch

    q.get()


def build_lm(model_config: ModelConfig, device='cuda') -> AbsDecoder:
    if isinstance(device, int):
        device = f'cuda:{device}'

    assert isinstance(model_config, ModelConfig)
    with open(model_config.f_config, 'r') as fi:
        configures = json.load(fi)
    model = lm_builder(None, configures, dist=False)
    model = utils.load_checkpoint(model.to(device), model_config.f_check)
    model = model.lm
    model.eval()
    return model


if __name__ == "__main__":
    parser = utils.BasicDDPParser(istraining=False)

    parser.add_argument("nbestlist", type=str,
                        help="Path to N-best list files.")
    parser.add_argument("output", type=str, help="The output text file. ")
    parser.add_argument("--lm-config", type=str,
                        help="Config of LM.")
    parser.add_argument("--lm-check", type=str,
                        help="Checkpoint of LM.")
    parser.add_argument("--lamb", type=float,
                        help="Setup the weight(s) of LM score.")
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")

    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true', default=False)

    args = parser.parse_args()

    main(args)
