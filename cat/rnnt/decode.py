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
    TestPadCollate,
    InferenceDistributedSampler
)

import re
import os
import json
import time
import uuid
import pickle
import argparse
import sentencepiece as spm
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

    cachedir = '/tmp'
    fmt = os.path.join(cachedir, str(uuid.uuid4())+r".{}.tmp")
    t_beg = time.time()
    if args.cpu:
        model, ext_lm = build_model(args, 'cpu')
        model.share_memory()
        if ext_lm is not None:
            ext_lm.share_memory()

        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, fmt, (model, ext_lm)))
    else:
        mp.spawn(main_worker, nprocs=world_size, args=(args, fmt))

    t_end = time.time()
    print("\nTime of searching: {:.2f}s".format(t_end-t_beg))

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


def main_worker(gpu: int, args: argparse.Namespace, fmt: str, models=None):

    args.gpu = gpu
    # only support one node
    args.rank = gpu
    world_size = args.world_size

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(1)
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

    testset = ScpDataset(args.input_scp)
    data_sampler = InferenceDistributedSampler(testset)
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, sampler=data_sampler, collate_fn=TestPadCollate())

    searcher = TransducerBeamSearcher(
        model.decoder, model.joint, blank_id=0, bos_id=model.bos_id, beam_size=args.beam_size,
        nbest=args.beam_size, algo=args.algo, prefix_merge=True, umax_portion=args.umax_portion,
        state_beam=2.3, expand_beam=2.3, temperature=1.0,
        lm_module=ext_lm, lm_weight=args.lm_weight)

    decode(args, model.encoder, searcher, testloader,
           device=device, local_writer=fmt.format(gpu))

    return None


@torch.no_grad()
def decode(args, encoder, searcher, testloader, device, local_writer):

    sp = spm.SentencePieceProcessor(model_file=args.spmodel)
    L = sum([1 for _ in testloader])
    nbest = {}
    cnt_frame = 0
    # t_beg = time.time()
    with autocast(enabled=(True if device != 'cpu' else False)), open(local_writer, 'w') as fi:
        for i, batch in enumerate(testloader):
            key, x, x_lens = batch
            key = key[0]
            x = x.to(device)
            enc_o, _ = encoder(x, x_lens)
            nbest_list, scores_nbest = searcher(
                enc_o)

            cnt_frame += enc_o.size(1)
            nbest[key] = [(score.item(), sp.decode(hypo))
                          for hypo, score in zip(nbest_list[0], scores_nbest[0])]
            _, best_seq = nbest[key][0]

            if args.lower:
                seq = best_seq.lower()
            else:
                seq = best_seq.upper()
            fi.write("{} {}\n".format(key, seq))
            print(
                "\r|{:<50}|[{:>5}/{:<5}]".format(int((i+1)/L*50)*'#', i+1, L), end='')

    with open(f"{local_writer}.nbest", 'wb') as fi:
        pickle.dump(nbest, fi)


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

    if args.lm_weight == 0.0 or args.ext_lm_config is None or args.ext_lm_check is None:
        return model, None
    else:
        assert args.ext_lm_check is not None

        with open(args.ext_lm_config, 'r') as fi:
            lm_configures = json.load(fi)
        ext_lm_model = lm_builder(args, lm_configures, dist=False)
        ext_lm_model = utils.load_checkpoint(
            ext_lm_model.to(device), args.ext_lm_check)
        ext_lm_model = ext_lm_model.lm
        ext_lm_model.eval()
        return model, ext_lm_model


def DecoderParser():

    parser = utils.BasicDDPParser(
        istraining=False, prog='RNN-Transducer decoding.')

    parser.add_argument("--ext-lm-config", type=str, default=None,
                        help="Config of external LM.")
    parser.add_argument("--ext-lm-check", type=str, default=None,
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
    parser.add_argument("--umax-portion", type=float,
                        default=0.35, help="Umax/T for ALSD decoding.")
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--lower", action='store_true', default=False)
    return parser


if __name__ == '__main__':
    parser = DecoderParser()
    args = parser.parse_args()
    main(args)
