"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan
"""

import os
import json
import utils
import argparse
import sentencepiece as spm
from tqdm import tqdm
from transducer_train import build_model, Transducer
from dataset import ScpDataset, TestPadCollate
from collections import OrderedDict

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def main(args):

    if not os.path.isfile(args.spmodel):
        raise FileNotFoundError(
            "Invalid sentencepiece model location: {}".format(args.spmodel))

    if not torch.cuda.is_available() or args.cpu:
        utils.highlight_msg("Using CPU.")
        single_worker('cpu', f"{args.output_dir}/decode.0.tmp", args)
        return None

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    if args.world_size == 1:
        single_worker('cuda:0', f"{args.output_dir}/decode.0.tmp", args)
        return None

    L_set = sum(1 for _ in open(args.input_scp, 'r'))
    res = L_set % args.world_size

    if res == 0:
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
        return None
    else:
        # This is a hack for non-divisible length of data to number of GPUs
        utils.highlight_msg("Using hack to deal with undivisible data length.")
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args, L_set-res))

        single_worker(
            "cuda:0", f"{args.output_dir}/decode.{args.world_size}.tmp", args, L_set-res)


def main_worker(gpu, ngpus_per_node, args, len_dataset: int = -1):
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    testset = ScpDataset(args.input_scp, idx_end=len_dataset)

    dist_sampler = DistributedSampler(testset)
    dist_sampler.set_epoch(1)

    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True,
        sampler=dist_sampler, collate_fn=TestPadCollate())

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures)

    if args.resume is not None:
        model = load_checkpoint(model, args.resume, loc=f'cuda:{args.gpu}')

    model.eval()

    decode(args, model.module, testloader, args.gpu,
           f"{args.output_dir}/decode.{args.rank}.tmp")


def single_worker(device, path_out, args, idx_beg=0):

    testset = ScpDataset(args.input_scp, idx_beg=idx_beg)

    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=TestPadCollate())

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures, dist=False)

    model = model.to(device)
    if args.resume is not None:
        model = load_checkpoint(model, args.resume, loc=device)

    model.eval()

    decode(args, model, testloader, device, path_out)


@torch.no_grad()
def decode(args, model: Transducer, testloader, device, local_writer):
    sp = spm.SentencePieceProcessor(model_file=args.spmodel)
    results = []
    for batch in tqdm(testloader):
        # for batch in testloader:
        key, x, x_lens = batch
        x = x.to(device, non_blocking=True)

        pred = model.decode(x, x_lens, mode=args.mode,
                            beam_size=args.beam_size)

        seq = sp.decode(pred.data.cpu().tolist())
        results.append((key, seq))

    with open(local_writer, 'w') as fi:
        for key, pred in results:
            assert len(key) == 1
            fi.write("{} {}\n".format(key[0], pred[0]))


def load_checkpoint(model: Transducer, path_ckpt, loc='cpu') -> Transducer:

    checkpoint = torch.load(path_ckpt, map_location=loc)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Logit calculation")
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of backbone.")
    parser.add_argument("--mode", type=str,
                        choices=['greedy', 'beam'], default='beam')
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:12947', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    args = parser.parse_args()

    main(args)
