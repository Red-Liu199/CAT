# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Rescoring with pre-trained GPT-2 model or custom LM.
NOTE (huahuan):
    For GPT-2 rescoring:
        At the first time of running the script, the model will be automatically downloaded.
        GPT-2 model is a cased model.

        References:
        https://huggingface.co/gpt2
"""

from ..shared import coreutils as utils
from . import lm_builder
from ..shared.data import NbestListDataset, NbestListCollate, InferenceDistributedSampler

import re
import os
import json
import time
import uuid
import argparse
import sentencepiece as sp

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


def main(args):
    if not args.gpt2:
        assert os.path.isfile(
            args.lm_config), f"File not found: {args.lm_config}"
        assert os.path.isfile(
            args.lm_check), f"File not found: {args.lm_check}"
    assert os.path.isfile(args.nbest), f"File not found: {args.nbest}"
    o_path = os.path.dirname(args.output)
    assert os.path.isdir(o_path), f"Directory not found: {o_path}"
    setattr(args, 'o_path', o_path)

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
    fmt = randomstr+r".{}.tmp"
    try:
        t_beg = time.time()
        if args.cpu:
            model = build_lm(args, 'cpu')
            model.share_memory()
            mp.spawn(main_worker, nprocs=world_size, args=(args, fmt, model))
        else:
            mp.spawn(main_worker, nprocs=world_size, args=(args, fmt))

        t_end = time.time()
        print("\nTime of rescoring: {:.2f}s".format(t_end-t_beg))
    except KeyboardInterrupt:
        print(" Stop rescoring...")
        for worker in range(world_size):
            os.remove(fmt.format(worker))
        exit(1)

    with open(args.output, 'w') as fo:
        for worker in range(world_size):
            path = fmt.format(worker)
            with open(path, 'r') as fi:
                fo.write(fi.read())
            os.remove(path)


@torch.no_grad()
def main_worker(gpu: int, args: argparse.Namespace, fmt: str = "rescore.{}.tmp", model=None):
    args.gpu = gpu
    args.rank = gpu
    world_size = args.world_size

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(1)
        dist.init_process_group(
            backend='gloo', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)
    else:
        device = gpu
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)

        model = build_lm(args, device)

    dataset = NbestListDataset(args.nbest)
    data_sampler = InferenceDistributedSampler(dataset)
    if args.gpt2:
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(e)
            print("'transformers' package is required for GPT-2 rescoring.")
            exit(1)
    else:
        assert args.spmodel is not None, "You need to specify --spmodel if not using pretrained GPT-2."
        assert os.path.isfile(args.spmodel), f"File not found: {args.spmodel}"
        tokenizer = sp.SentencePieceProcessor(model_file=args.spmodel)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                            sampler=data_sampler, collate_fn=NbestListCollate(tokenizer, isGPT=args.gpt2))
    writer = os.path.join(args.o_path, fmt.format(gpu))
    # rescoring
    with autocast(enabled=(True if device != 'cpu' else False)), open(writer, 'w') as fo:
        for i, batch in enumerate(dataloader):
            rescored_list = {}
            keys, texts, scores, tokens = batch
            tokens['input_ids'] = tokens['input_ids'].to(device)
            input, mask = tokens['input_ids'], tokens['attention_mask']
            if args.gpt2:
                logits = model(**tokens)['logits']
                mask = 1 - mask
            else:
                logits, _ = model(input)

            log_p = logits.log_softmax(dim=-1)
            ll = log_p.gather(dim=-1, index=input.unsqueeze(-1)
                              ).squeeze(-1)    # type:torch.Tensor
            ll = ll.masked_fill_(mask, float(0.0))

            # length norm
            ll = ll.sum(dim=-1) / (input.size(1) - mask.sum(dim=-1))

            ########## DEBUG CODE ###########
            # print("AC score")
            # print(scores)
            # print("LM score")
            # print(ll)
            # raise KeyboardInterrupt
            #################################

            for k, t, ac_score, lm_score in zip(keys, texts, scores, ll.cpu()):
                new_score = ac_score + args.lamb * lm_score
                if k not in rescored_list:
                    rescored_list[k] = (new_score, t)
                else:
                    if new_score > rescored_list[k][0]:
                        rescored_list[k] = (new_score, t)

            for key, (score, seq) in rescored_list.items():
                if args.gpt2:
                    seq = seq.upper()
                fo.write(f"{key} {seq}\n")
            print("\r[{:>2}] {:>5}".format(args.gpu, i+1), end='')


def build_lm(args, device='cuda'):
    if isinstance(device, int):
        device = f'cuda:{device}'

    if args.gpt2:
        try:
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained('gpt2')
        except Exception as e:
            print(e)
            print("'transformers' package is required for GPT-2 rescoring.")
            exit(1)
    else:
        with open(args.lm_config, 'r') as fi:
            configures = json.load(fi)
        model = lm_builder(None, configures, dist=False)
        model = utils.load_checkpoint(model.to(device), args.lm_check)
        model = model.lm

    model.eval()
    return model


if __name__ == "__main__":
    parser = utils.BasicDDPParser(istraining=False)

    parser.add_argument("nbest", type=str, help="N-best list files.")
    parser.add_argument("output", type=str, help="The output text file. ")
    parser.add_argument("--lm-config", type=str, default=None,
                        help="Config of external LM.")
    parser.add_argument("--lm-check", type=str, default=None,
                        help="Checkpoint of external LM.")
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")

    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--lamb", type=float, default=0.005,
                        help="Setup the weight of LM score.")
    parser.add_argument("--gpt2", action="store_true", required=False,
                        help="Use GPT-2 pre-trained model.")

    args = parser.parse_args()

    main(args)
