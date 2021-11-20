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
from ..shared.data import (
    ScpDataset,
    TestPadCollate
)
from ..shared.decoder import NGram

import os
import json
import time
import uuid
import pickle
import argparse
import sentencepiece as spm
from tqdm import tqdm
from typing import Union, Tuple

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


def main(args):

    if not os.path.isfile(args.spmodel):
        raise FileNotFoundError(
            "Invalid sentencepiece model location: {}".format(args.spmodel))
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

    if args.rescore and args.lm_weight <= 0.0:
        raise RuntimeError(
            f"Trying to do rescoring with lm-weight={args.lm_weight}")

    cachedir = '/tmp'
    fmt = os.path.join(cachedir, str(uuid.uuid4())+r".{}.tmp")

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        print(re)
    q = mp.Queue(maxsize=world_size*2)
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

    for i in range(args.world_size*2):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print("\nTime = {:.2f} s | RTF = {:.2f} ".format(
        t_dur, t_dur*args.world_size / n_frames * 100))
    time.sleep(2)


def main_worker(gpu: int, args: argparse.Namespace, q: mp.Queue, fmt: str, models=None):
    if gpu == args.world_size:
        return dataserver(args, q)
    args.gpu = gpu
    # only support one node
    args.rank = gpu
    world_size = args.world_size

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(args.thread_per_woker)
        dist.init_process_group(
            backend='gloo', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)

        model, ext_lm = models
    else:
        device = gpu
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)

        model, ext_lm = build_model(args, device)

    searcher = TransducerBeamSearcher(
        model.decoder, model.joint, blank_id=0, bos_id=model.bos_id, beam_size=args.beam_size,
        nbest=args.beam_size, algo=args.algo, prefix_merge=True, umax_portion=args.umax_portion,
        state_beam=2.3, expand_beam=2.3, temperature=1.0, word_prefix_tree=args.word_tree,
        lm_module=ext_lm, lm_weight=args.lm_weight, rescore=args.rescore)

    local_writer = fmt.format(gpu)
    sp = spm.SentencePieceProcessor(model_file=args.spmodel)
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
            nbest[key] = [(score.item(), sp.decode(hypo))
                          for hypo, score in zip(nbest_list[0], scores_nbest[0])]
            _, best_seq = nbest[key][0]
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

    if args.lm_weight == 0.0 or args.lm_config is None or args.lm_check is None:
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
        istraining=False, prog='RNN-Transducer decoding.')

    parser.add_argument("--lm-config", type=str, default=None,
                        help="Config of external LM.")
    parser.add_argument("--lm-check", type=str, default=None,
                        help="Checkpoint of external LM.")
    parser.add_argument("--lm-weight", type=float, default=0.0,
                        help="Weight of external LM.")

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default='./decode')
    parser.add_argument("--algo", type=str,
                        choices=['default', 'lc'], default='default')
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")
    parser.add_argument("--nj", type=int, default=None)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument("--umax-portion", type=float,
                        default=0.35, help="Umax/T for ALSD decoding.")
    parser.add_argument("--word-tree", type=str, default=None,
                        help="Path to word prefix tree file.")
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--rescore", action='store_true', default=False)
    return parser


if __name__ == '__main__':
    parser = DecoderParser()
    args = parser.parse_args()
    main(args)
