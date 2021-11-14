# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Rescoring with pre-trained GPT-2 model or custom LM.

This script is ported from cat.rnnt.decode

Rescore N-best list

when using two LM:
log(P_ac) + lamb1 * log(P_am1) - lamb2 * log(P_am2)
where P_am1 is external LM and P_am2 is internal LM.

when using only one LM:
w/ --ILM, log(P_ac) - lamb2 * log(P_am2)
w/o --ILM, log(P_ac) + lamb1 * log(P_am1)

GPT-2 couldn't be the internal LM.
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

import os
import json
import time
import uuid
import argparse
import sentencepiece as sp
from typing import Optional, Union, List


from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch


class ModelConfig:
    def __init__(self, f_config: Optional[str] = None, f_check: Optional[str] = None, use_gpt2: bool = False) -> None:
        self.isgpt2 = use_gpt2
        if f_config is not None:
            assert os.path.isfile(f_config)
        self.f_config = f_config
        if f_check is not None:
            assert os.path.isfile(f_check)
        self.f_check = f_check


def main(args):
    assert os.path.isfile(args.nbest), f"File not found: {args.nbest}"

    configs = args.lm_config
    checks = args.lm_check
    lambs = args.lamb
    assert len(lambs) >= 1
    for _lamb in lambs:
        assert _lamb >= 0.0, f"Consider using --ILM instead of negative --lamb={_lamb}"

    assert len(configs) <= 2
    assert len(checks) == len(checks)

    model_configs = []  # type: List[ModelConfig]
    if len(configs) == 0:
        assert args.gpt2 and not args.ILM, "You MUST specify at least one LM or --GPT2 w/o --ILM."
        model_configs.append(ModelConfig(use_gpt2=True))
    elif len(configs) == 1:
        # one LM
        assert len(lambs) == 1, f"--lamb={lambs} number mismatch with config"
        if args.ILM:
            assert not args.gpt2, "--gpt2 is conflict with --ILM in one LM mode."
            model_configs.append(ModelConfig(configs[0], checks[0]))
            lambs[0] = -lambs[0]
        else:
            if args.gpt2:
                model_configs.append(ModelConfig(use_gpt2=True))
            else:
                model_configs.append(ModelConfig(configs[0], checks[0]))
    else:
        assert not args.gpt2, "--gpt2 is conflict with two LMs mode."
        model_configs.append(ModelConfig(configs[0], checks[0]))
        lambs[1] = -lambs[1]
        model_configs.append(ModelConfig(configs[1], checks[1]))
    args.lamb = lambs

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

    t_beg = time.time()
    if args.cpu:
        models = build_lm(model_configs, 'cpu')
        for m in models:
            m.share_memory()
        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, fmt, None, models))
    else:
        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, fmt, model_configs))
    t_end = time.time()
    print("\nTime of rescoring: {:.2f}s".format(t_end-t_beg))

    with open(args.output, 'w') as fo:
        for worker in range(world_size):
            path = fmt.format(worker)
            with open(path, 'r') as fi:
                fo.write(fi.read())
            os.remove(path)


@torch.no_grad()
def main_worker(pid: int, args: argparse.Namespace, fmt: str = "rescore.{}.tmp", model_configs: Optional[List[ModelConfig]] = None, models=None):
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

    if models is None:
        models = build_lm(model_configs, 'cpu')

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
    writer = fmt.format(pid)
    # rescoring
    with autocast(enabled=(True if device != 'cpu' else False)), open(writer, 'w') as fo:
        for i, batch in enumerate(dataloader):
            rescored_list = {}
            keys, texts, scores, tokens = batch
            tokens['input_ids'] = tokens['input_ids'].to(device)
            input, mask = tokens['input_ids'], tokens['attention_mask']
            if args.gpt2:
                mask = 1 - mask

            log_p_lm = 0.
            for _m, _lamb in zip(models, args.lamb):
                if args.gpt2:
                    logits = _m(**tokens)['logits']
                else:
                    logits, _ = _m(input)

                log_p = logits.log_softmax(dim=-1)
                ll = log_p.gather(dim=-1, index=input.unsqueeze(-1)
                                  ).squeeze(-1)    # type:torch.Tensor
                ll = ll.masked_fill_(mask, float(0.0))
                log_p_lm += _lamb * ll.sum(dim=-1)

            # length norm
            log_p_lm /= input.size(1) - mask.sum(dim=-1)

            ########## DEBUG CODE ###########
            # print("AC score")
            # print(scores)
            # print("LM score")
            # print(log_p_lm)
            # raise KeyboardInterrupt
            #################################

            for k, t, ac_score, lm_score in zip(keys, texts, scores, log_p_lm.cpu()):
                new_score = ac_score + lm_score
                if k not in rescored_list:
                    rescored_list[k] = (new_score, t)
                else:
                    if new_score > rescored_list[k][0]:
                        rescored_list[k] = (new_score, t)

            for key, (score, seq) in rescored_list.items():
                fo.write(f"{key} {seq}\n")
            print("\r[{:>2}] {:>5}".format(args.pid, i+1), end='')


def build_lm(model_configs: List[ModelConfig], device='cuda') -> List:
    if isinstance(device, int):
        device = f'cuda:{device}'

    output = []
    for _config in model_configs:
        assert isinstance(_config, ModelConfig)
        if _config.isgpt2:
            try:
                from transformers import GPT2LMHeadModel
                model = GPT2LMHeadModel.from_pretrained('gpt2')
            except Exception as e:
                print(e)
                print("'transformers' package is required for GPT-2 rescoring.")
                exit(1)
        else:
            with open(_config.f_config, 'r') as fi:
                configures = json.load(fi)
            model = lm_builder(None, configures, dist=False)
            model = utils.load_checkpoint(model.to(device), _config.f_check)
            model = model.lm
        model.eval()
        output.append(model)
    return output


if __name__ == "__main__":
    parser = utils.BasicDDPParser(istraining=False)

    parser.add_argument("nbest", type=str, help="N-best list files.")
    parser.add_argument("output", type=str, help="The output text file. ")
    parser.add_argument("--lm-config", type=str, nargs='*',
                        help="Config of internal/external LM(s).")
    parser.add_argument("--lm-check", type=str, nargs='*',
                        help="Checkpoint of internal/external LM(s).")
    parser.add_argument("--lamb", type=float, nargs='+',
                        help="Setup the weight(s) of LM score.")
    parser.add_argument("--ILM", action='store_true', default=False,
                        help="When using one LM, set that as internal LM instead of external one.")
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")

    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--gpt2", action="store_true", required=False,
                        help="Use GPT-2 pre-trained model.")

    args = parser.parse_args()

    main(args)
