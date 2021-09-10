"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

Parallel decode with multi-gpu and single-gpu-multi-process support 
"""

import coreutils
from lm_train import build_model as lm_build
from dataset import ScpDataset, TestPadCollate
from transducer_train import build_model, Transducer, ConvJointNet
from beam_search_base import BeamSearchRNNTransducer, BeamSearchConvTransducer, ConvMemBuffer
from beam_search_transducer import TransducerBeamSearcher

import os
import json
import argparse
import sentencepiece as spm
from tqdm import tqdm
from collections import OrderedDict
from typing import Union, List, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def main(args):

    if not os.path.isfile(args.spmodel):
        raise FileNotFoundError(
            "Invalid sentencepiece model location: {}".format(args.spmodel))

    if not torch.cuda.is_available() or args.cpu:
        coreutils.highlight_msg("Using CPU")
        single_worker(args, 'cpu')
        return None

    world_size = torch.cuda.device_count() * args.nj

    # L_set = sum(1 for _ in open(args.input_scp, 'r'))
    # indices = equalSplitIdx(L_set, world_size)
    indices = equalLenSplit(args.input_scp, world_size)

    mp.spawn(main_worker, nprocs=world_size,
             args=(args, indices))


def equalSplitIdx(tot_len: int, N: int, idx_beg=0, idx_end=-1):
    if idx_end == -1:
        idx_end = tot_len
    interval = tot_len // N
    indices = [interval * i + idx_beg for i in range(N+1)]
    indices[-1] = idx_end
    return indices


def equalLenSplit(scp_in: str, N: int, idx_beg=0, idx_end=-1):

    testset = ScpDataset(scp_in, idx_beg=idx_beg, idx_end=idx_end, sort=True)

    if idx_end == -1:
        idx_end = len(testset)

    L = idx_end - idx_beg
    if L < N:
        raise RuntimeError(f"len(set) < N: {L} < {N}")
    cnt = 0
    for i in range(L):
        key, mat = testset[i]
        cnt += mat.size(0)

    avg = float(cnt) / N

    # greedy not optimal
    indices = [idx_beg]
    cnt_interval = 0
    for i in range(L):
        key, mat = testset[i]
        cnt_interval += mat.size(0)
        if cnt_interval >= avg:
            indices.append(i+idx_beg)
            cnt_interval = 0

    while len(indices) < N+1:
        indices[1:] = [x-1 for x in indices[1:]]
        indices.append(idx_end)

    indices[-1] = idx_end
    return indices


def main_worker(rank: int, args: argparse.Namespace, intervals: List[int]):

    gpu = rank // args.nj
    single_worker(
        args, gpu, idx_beg=intervals[rank], idx_end=intervals[rank+1], suffix='{}-{}'.format(gpu, rank))
    return


def single_worker(args: argparse.Namespace, device: Union[int, str], idx_beg: int = 0, idx_end: int = -1, suffix: str = '0-0'):

    if device != 'cpu':
        torch.cuda.set_device(device)

    if args.ext_lm_config is None:
        model = gen_model(args, device)
    else:
        model, ext_lm = gen_model(args, device, use_ext_lm=True)
        ext_lm.eval()

    model.eval()

    testset = ScpDataset(args.input_scp, idx_beg=idx_beg, idx_end=idx_end)
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=TestPadCollate())

    writer = os.path.join(args.output_dir, f'decode.{suffix}.tmp')
    decode(args, model, testloader, device=device, local_writer=writer)


@torch.no_grad()
def decode(args, model: Transducer, testloader, device, local_writer):
    sp = spm.SentencePieceProcessor(model_file=args.spmodel)
    if args.mode == 'beam':
        # if isinstance(model.joint, ConvJointNet):
        #     beamsearcher = BeamSearchConvTransducer(
        #         model, kernel_size=(3, 3), beam_size=args.beam_size)
        # else:
        #     beamsearcher = BeamSearchRNNTransducer(
        #         model, beam_size=args.beam_size)
        # beamsearcher = beamsearcher.to(device)
        beamsearcher = TransducerBeamSearcher(model, 0, args.beam_size)
    else:
        beamsearcher = None
    results = []
    L = len(testloader)
    for i, batch in enumerate(testloader):
        key, x, x_lens = batch
        x = x.to(device)

        pred, _, _, _ = model.decode(x, x_lens, mode=args.mode,
                                     beamSearcher=beamsearcher)

        seq = sp.decode(pred[0])
        results.append((key, seq))
        print(
            "\r|{:<80}|[{:>5}/{:<5}]".format(int((i+1)/L*80)*'#', i+1, L), end='')
    print("")
    with open(local_writer, 'w') as fi:
        for key, pred in results:
            assert len(key) == 1
            fi.write("{} {}\n".format(key[0], pred[0]))


def gen_model(args, device, use_ext_lm=False) -> Union[Tuple[torch.nn.Module, torch.nn.Module], torch.nn.Module]:
    if isinstance(device, int):
        device = f'cuda:{device}'

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures, dist=False, verbose=False)
    model = model.to(device)
    assert args.resume is not None, "Trying to decode with uninitialized parameters. Add --resume"

    model = load_checkpoint(model, args.resume)

    if use_ext_lm:
        assert args.ext_lm_check is not None

        with open(args.ext_lm_config, 'r') as fi:
            lm_configures = json.load(fi)
        ext_lm_model = lm_build(args, lm_configures, dist=False)
        ext_lm_model = load_checkpoint(
            ext_lm_model.to(device), args.ext_lm_check)
        return model, ext_lm_model
    else:
        return model


def load_checkpoint(model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel], path_ckpt: str) -> Transducer:

    checkpoint = torch.load(
        path_ckpt, map_location=next(model.parameters()).device)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = checkpoint['model']
    else:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            # remove the 'module.'
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':

    parser = coreutils.BasicDDPParser(istraining=False)

    parser.add_argument("--ext-lm-config", type=str, default=None,
                        help="Config of external LM.")
    parser.add_argument("--ext-lm-check", type=str, default=None,
                        help="Checkpoint of external LM.")

    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--mode", type=str,
                        choices=['greedy', 'beam'], default='beam')
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")
    parser.add_argument("--nj", type=int, default=2)
    parser.add_argument("--cpu", action='store_true', default=False)

    args = parser.parse_args()

    main(args)


# TODO: sort the sequence by length then feed into worker with equal total lengths.
