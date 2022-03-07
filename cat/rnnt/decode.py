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
        if not args.silience:
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
        if not args.silience:
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

    q_data_producer = mp.Queue(maxsize=world_size)
    q_nbest_saver = mp.Queue(maxsize=world_size)
    if args.cpu:
        model, ext_lm = build_model(args, 'cpu')
        model.share_memory()
        if ext_lm is not None:
            ext_lm.share_memory()

        mp.spawn(main_worker, nprocs=world_size+2,
                 args=(args, q_data_producer, q_nbest_saver, fmt, (model, ext_lm)))
    else:
        mp.spawn(main_worker, nprocs=world_size+2,
                 args=(args, q_data_producer, q_nbest_saver, fmt))

    del q_nbest_saver
    del q_data_producer


def dataserver(args, q: mp.Queue):
    testset = ScpDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=0,
        collate_fn=TestPadCollate())

    f_nbest = args.output_prefix+'.nbest'
    if os.path.isfile(f_nbest):
        with open(f_nbest, 'rb') as fi:
            nbest = pickle.load(fi)
    else:
        nbest = {}

    t_beg = time.time()
    for batch in tqdm(testloader, total=len(testloader), disable=(args.silience)):
        key = batch[0][0]
        """
        NOTE: 
        In some cases (decoding with large beam size or large LMs like Transformer), 
        ... the decoding consumes too much memory and would probably causes OOV error.
        So I add checkpointing output in the nbest list. However, things would be
        ... complicated when first decoding w/o --estimate-ILM, save the checkpoint to nbest list
        ... then continue decoding w/ --estimate-ILM.
        I just assume users won't do that.
        """
        if key not in nbest:
            q.put(batch, block=True)

    for i in range(args.world_size*2):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    if not args.silience:
        print("\nTime = {:.2f} s | RTF = {:.2f} ".format(
            t_dur, t_dur*args.world_size / n_frames * 100))
    time.sleep(2)


def consumer_output(args, q: mp.Queue):
    """Get data from queue and save to file."""
    def load_and_save(_nbest: dict):
        if os.path.isfile(f_nbest):
            with open(f_nbest, 'rb') as fi:
                _nbest.update(pickle.load(fi))
        with open(f_nbest, 'wb') as fo:
            pickle.dump(_nbest, fo)

    f_nbest = args.output_prefix+'.nbest'
    interval_check = 1000   # save nbestlist to file every 1000 steps
    cnt_done = 0
    nbest = {}
    while True:
        nbestlist = q.get(block=True)
        if nbestlist is None:
            cnt_done += 1
            if cnt_done == args.world_size:
                break
            continue
        nbest.update(nbestlist)
        del nbestlist
        if len(nbest) % interval_check == 0:
            load_and_save(nbest)
            nbest = {}

    load_and_save(nbest)
    # write the 1-best result to text file.
    with open(args.output_prefix, 'w') as fo:
        for k, hypo_items in nbest.items():
            if args.estimate_ILM and k[-4:] == "-ilm":
                continue
            best_hypo = max(hypo_items.values(), key=lambda item: item[0])[1]
            fo.write(f"{k} {best_hypo}\n")

    del load_and_save


def main_worker(pid: int, args: argparse.Namespace, q_data: mp.Queue, q_nbest: mp.Queue, fmt: str, models=None):
    if pid == args.world_size:
        return dataserver(args, q_data)
    elif pid == args.world_size + 1:
        return consumer_output(args, q_nbest)

    args.gpu = pid
    # only support one node
    args.rank = pid

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(args.thread_per_woker)
        model, ext_lm = models
    else:
        device = pid
        torch.cuda.set_device(device)
        model, ext_lm = build_model(args, device)

    est_ilm = args.estimate_ILM
    searcher = TransducerBeamSearcher(
        decoder=model.decoder, joint=model.joint,
        blank_id=0, bos_id=model.bos_id, beam_size=args.beam_size,
        nbest=args.beam_size, algo=args.algo, umax_portion=args.umax_portion,
        prefix_merge=True, lm_module=ext_lm, alpha=args.alpha, beta=args.beta,
        word_prefix_tree=args.word_tree, rescore=args.rescore, est_ilm=est_ilm, verbose=(pid == 0))

    local_writer = fmt.format(pid)
    tokenizer = tknz.load(args.tokenizer)
    nbest = {}
    with torch.no_grad(), autocast(enabled=(True if device != 'cpu' else False)), open(local_writer, 'w') as fi:
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_lens = batch
            key = key[0]
            x = x.to(device)
            nbest_list, scores_nbest = searcher(model.encoder(x, x_lens)[0])
            nbest[key] = {
                bid: (score.item(), tokenizer.decode(hypo.cpu().tolist()))
                for bid, (score, hypo) in enumerate(zip(scores_nbest, nbest_list))
            }
            if est_ilm:
                nbest[key+'-ilm'] = {
                    bid: (searcher.ilm_score[bid].item(), trans) for bid, (_, trans) in nbest[key].items()
                }
            q_nbest.put(nbest, block=True)
            nbest = {}
            del batch

    q_nbest.put(None)
    q_data.get()


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
                        choices=['default', 'lc', 'alsd', 'rna'], default='default')
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model location. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument("--umax-portion", type=float,
                        default=0.35, help="Umax/T for ALSD decoding.")
    parser.add_argument("--estimate-ILM", action="store_true",
                        help="Enable internal language model estimation. "
                        "This would slightly slow down the decoding. "
                        "With this option, the N-best list file would contains ILM scores too. See utils/dispatch_ilm.py")
    parser.add_argument("--word-tree", type=str, default=None,
                        help="Path to word prefix tree file.")
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--rescore", action='store_true', default=False)
    parser.add_argument("--silience", action='store_true', default=False)
    return parser


if __name__ == '__main__':
    parser = DecoderParser()
    args = parser.parse_args()
    main(args)
