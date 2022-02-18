# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""
Parallel decode with distributed GPU/CPU support 
"""

from ..lm import lm_builder
from . import rnnt_builder
from .beam_search_transducer import TransducerBeamSearcher
from ..shared import coreutils as utils
from ..shared import tokenizer as tknz
from ..shared.data import (
    ScpDataset,
    TestPadCollate
)

import os
import json
import time
import pickle
import argparse
from tqdm import tqdm
from typing import Union, Tuple

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


def main(args):

    if args.tokenizer is None or not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(
            "Invalid tokenizer model location: {}".format(args.tokenizer))
    if not torch.cuda.is_available() or args.cpu:
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

    if args.rescore and args.alpha is None:
        print("WARNING: "
              f"trying to rescore with alpha not specified.\n"
              "set rescore=False")
        args.rescore = False

    cachedir = '/tmp'
    assert os.path.isdir(cachedir), f"Cache directory not found: {cachedir}"
    if not os.access(cachedir, os.W_OK):
        raise PermissionError(f"Permission denied for writing to {cachedir}")
    fmt = os.path.join(cachedir, utils.gen_random_string()+r".{}.tmp")

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        print(re)
    q = mp.Queue(maxsize=world_size)
    if args.cpu:
        model, ext_lm = build_model(args, 'cpu')
        model.share_memory()
        if ext_lm is not None:
            ext_lm.share_memory()

        mp.spawn(main_worker, nprocs=world_size+1,
                 args=(args, q, fmt, (model, ext_lm)))
    else:
        mp.spawn(main_worker, nprocs=world_size+1, args=(args, q, fmt))

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
    # sort the dataset in desencding order
    testset_ls = testset.get_seq_len()
    len_match = sorted(list(zip(testset_ls, testset._dataset)),
                       key=lambda item: item[0])
    testset._dataset = [data for _, data in len_match]
    n_frames = sum(testset_ls)
    del len_match, testset_ls
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

    for i in range(args.world_size*2):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print("\nTime = {:.2f} s | RTF = {:.2f} ".format(
        t_dur, t_dur*args.world_size / n_frames * 100))
    time.sleep(2)


def main_worker(pid: int, args: argparse.Namespace, q: mp.Queue, fmt: str, models=None):
    if pid == args.world_size:
        return dataserver(args, q)
    args.gpu = pid
    # only support one node
    args.rank = pid
    world_size = args.world_size

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(args.thread_per_woker)
        model, ext_lm = models
    else:
        device = pid
        torch.cuda.set_device(device)
        model, ext_lm = build_model(args, device)

    searcher = TransducerBeamSearcher(
        decoder=model.decoder, joint=model.joint,
        blank_id=0, bos_id=model.bos_id, beam_size=args.beam_size,
        nbest=args.beam_size, algo=args.algo, umax_portion=args.umax_portion,
        prefix_merge=True, lm_module=ext_lm, alpha=args.alpha, beta=args.beta,
        state_beam=2.3, expand_beam=2.3, temperature=1.0,
        word_prefix_tree=args.word_tree, rescore=args.rescore, verbose=(pid == 0))

    local_writer = fmt.format(pid)
    tokenizer = tknz.load(args.tokenizer)
    nbest = {}
    with torch.no_grad(), autocast(enabled=(True if device != 'cpu' else False)), open(local_writer, 'w') as fi:
        while True:
            batch = q.get(block=True)
            if batch is None:
                break
            key, x, x_lens = batch
            key = key[0]
            x = x.to(device)
            nbest_list, scores_nbest = searcher(model.encoder(x, x_lens)[0])
            nbest_list = [pred.cpu().tolist() for pred in nbest_list]
            if args.token_nbest:
                nbest[key] = [(score.item(), hypo)
                              for hypo, score in zip(nbest_list, scores_nbest)]
            else:
                nbest[key] = [(score.item(), tokenizer.decode(hypo))
                              for hypo, score in zip(nbest_list, scores_nbest)]
            best_seq = tokenizer.decode(nbest_list[0])
            fi.write("{} {}\n".format(key, best_seq))
            del batch

    with open(f"{local_writer}.nbest", 'wb') as fi:
        pickle.dump(nbest, fi)
    q.get()


def build_model(args, device) -> Tuple[torch.nn.Module, Union[torch.nn.Module, None]]:
    if isinstance(device, int):
        device = f'cuda:{device}'

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = rnnt_builder(args, configures, dist=False, verbose=False)
    model = model.to(device)
    assert args.resume is not None, "Trying to decode with uninitialized parameters. Add --resume"

    model = utils.load_checkpoint(model, args.resume)
    model.eval()

    if args.alpha is None or args.lm_config is None or args.lm_check is None:
        return model, None
    else:
        assert args.lm_check is not None

        with open(args.lm_config, 'r') as fi:
            lm_configures = json.load(fi)
        ext_lm_model = lm_builder(args, lm_configures, dist=False)
        if lm_configures['decoder']['type'] != "NGram":
            ext_lm_model = utils.load_checkpoint(
                ext_lm_model.to(device), args.lm_check)
        ext_lm_model = ext_lm_model.lm
        ext_lm_model.eval()
        return model, ext_lm_model


def DecoderParser():

    parser = utils.BasicDDPParser(
        istraining=False, prog='RNN-Transducer decoder.')

    parser.add_argument("--lm-config", type=str, default=None,
                        help="Config of external LM.")
    parser.add_argument("--lm-check", type=str, default=None,
                        help="Checkpoint of external LM.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Weight of external LM.")
    parser.add_argument("--beta", type=float, default=None,
                        help="Penalty value of external LM.")

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default='./decode')
    parser.add_argument("--algo", type=str,
                        choices=['default', 'lc'], default='default')
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model location. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument("--umax-portion", type=float,
                        default=0.35, help="Umax/T for ALSD decoding.")
    parser.add_argument("--word-tree", type=str, default=None,
                        help="Path to word prefix tree file.")
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--rescore", action='store_true', default=False)
    parser.add_argument("--token-nbest", action="store_true",
                        help="Store N-best list in tokens intead of text.")
    return parser


if __name__ == '__main__':
    parser = DecoderParser()
    args = parser.parse_args()
    main(args)
