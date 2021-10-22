"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

Parallel decode with multi-gpu and single-gpu-multi-process support 
"""

import coreutils
from lm_train import build_model as lm_build
from data import ScpDataset, TestPadCollate, InferenceDistributedSampler
from transducer_train import build_model
from beam_search_transducer import TransducerBeamSearcher

import re
import os
import json
import pickle
import argparse
import sentencepiece as spm
from collections import OrderedDict
from typing import Union, List, Tuple

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

    # generate encoder output first
    prefix = os.path.basename(args.input_scp).split('.')[0]
    binary_enc = os.path.join(
        args.enc_out_dir, f"{prefix}.bin")
    link_enc = os.path.join(
        args.enc_out_dir, f"{prefix}.lnk")
    if not os.path.isfile(binary_enc) or not os.path.isfile(link_enc):
        print("> Encoder output file not found, generating...")
        compute_encoder_output(args, enc_bin=binary_enc, enc_link=link_enc)
    setattr(args, 'enc_bin', binary_enc)
    setattr(args, 'enc_lnk', link_enc)

    mp.spawn(main_worker, nprocs=world_size, args=(args,))

    nbest_pattern = re.compile(r'[.]nbest$')
    with open(os.path.join(args.output_dir, 'nbest.pkl'), 'wb') as fo:
        all_nbest = {}
        for nbest_f in os.listdir(args.output_dir):
            if nbest_pattern.search(nbest_f) is None:
                continue

            partial_bin = os.path.join(args.output_dir, nbest_f)
            with open(partial_bin, 'rb') as fi:
                partial_nbest = pickle.load(fi)  # type: dict

            all_nbest.update(partial_nbest)
            os.remove(partial_bin)
        pickle.dump(all_nbest, fo)


def main_worker(gpu: int, args: argparse.Namespace):

    args.gpu = gpu
    # only support one node
    args.rank = gpu
    world_size = args.world_size

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads((os.cpu_count() * 2)//world_size)
        dist.init_process_group(
            backend='gloo', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)
    else:
        device = gpu
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=world_size, rank=args.rank)

    if device != 'cpu':
        torch.cuda.set_device(device)

    if args.lm_weight == 0.0 or args.ext_lm_config is None or args.ext_lm_check is None:
        model = gen_model(args, device)
        ext_lm = None
    else:
        model, ext_lm = gen_model(args, device, use_ext_lm=True)

    testset = ScpDataset(args.input_scp)
    data_sampler = InferenceDistributedSampler(testset)
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, sampler=data_sampler, collate_fn=TestPadCollate())

    writer = os.path.join(args.output_dir, f'decode.{gpu}.tmp')
    beamsearcher = TransducerBeamSearcher(
        model.decoder, model.joint, blank_id=0, bos_id=model.bos_id, beam_size=args.beam_size,
        nbest=5, algo='default',
        state_beam=2.3, expand_beam=2.3, lm_module=ext_lm, lm_weight=args.lm_weight)

    decode(args, beamsearcher, testloader,
           device=device, local_writer=writer)

    return None


@torch.no_grad()
def decode(args, beamsearcher, testloader, device, local_writer):
    f_enc_hid = open(args.enc_bin, 'rb')
    with open(args.enc_lnk, 'rb') as fi:
        f_enc_seeks = pickle.load(fi)

    def _load_enc_mat(k: str):
        f_enc_hid.seek(f_enc_seeks[k])
        return (pickle.load(f_enc_hid)).to(device, non_blocking=True)

    sp = spm.SentencePieceProcessor(model_file=args.spmodel)

    L = sum([1 for _ in testloader])
    nbest = {}
    with autocast(enabled=(True if device != 'cpu' else False)), open(local_writer, 'w') as fi:
        for i, batch in enumerate(testloader):
            key = batch[0][0]
            enc_o = _load_enc_mat(key)

            best_hypo, score_best_hypo, nbest_list, scores_nbest = beamsearcher(
                enc_o)

            nbest[key] = [(score.item(), sp.decode(hypo))
                          for hypo, score in zip(nbest_list[0], scores_nbest[0])]

            _, best_seq = nbest[key][0]

            if args.lower:
                seq = best_seq.lower()
            else:
                seq = best_seq.upper()
            fi.write("{} {}\n".format(key, seq))
            print(
                "\r|{:<60}|[{:>5}/{:<5}]".format(int((i+1)/L*60)*'#', i+1, L), end='')

    print("\r|{0}|[{1:>5}/{1:<5}]".format(60*'#', L))

    with open(f"{local_writer}.nbest", 'wb') as fi:
        pickle.dump(nbest, fi)

    f_enc_hid.close()


def gen_model(args, device, use_ext_lm=False) -> Union[Tuple[torch.nn.Module, torch.nn.Module], torch.nn.Module]:
    if isinstance(device, int):
        device = f'cuda:{device}'

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures, dist=False, verbose=False)
    model = model.to(device)
    assert args.resume is not None, "Trying to decode with uninitialized parameters. Add --resume"

    model = load_checkpoint(model, args.resume)
    model.eval()

    if use_ext_lm:
        assert args.ext_lm_check is not None

        with open(args.ext_lm_config, 'r') as fi:
            lm_configures = json.load(fi)
        ext_lm_model = lm_build(args, lm_configures, dist=False)
        ext_lm_model = load_checkpoint(
            ext_lm_model.to(device), args.ext_lm_check)
        ext_lm_model = ext_lm_model.lm
        ext_lm_model.eval()
        return model, ext_lm_model
    else:
        return model


def compute_encoder_output(args, enc_bin: str, enc_link: str):
    if not torch.cuda.is_available() or args.cpu:
        usegpu = False
    else:
        usegpu = True

    num_workers = args.world_size
    # binary files saved as f'{enc_bin}.x'
    mp.spawn(worker_compute_enc_out, nprocs=num_workers,
             args=(num_workers, enc_bin, usegpu, args))

    print("> Merging sub-process output files..")
    fseeks = {}
    with open(enc_bin, 'wb') as fo:
        for i in range(num_workers):
            with open(f'{enc_bin}.{i}', 'rb') as fi:
                # type: List[Tuple[str, torch.Tensor]]
                part_data = pickle.load(fi)

            for key, mat in part_data:
                fseeks[key] = fo.tell()
                pickle.dump(mat, fo)
            os.remove(f'{enc_bin}.{i}')

    with open(enc_link, 'wb') as fo:
        pickle.dump(fseeks, fo)
    print("  files merging done.")
    torch.cuda.empty_cache()


@torch.no_grad()
def worker_compute_enc_out(gpu: int, world_size: int, suffix: str, usegpu: bool, args):
    rank = gpu

    if usegpu:
        device = gpu
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=world_size, rank=rank)
    else:
        device = 'cpu'
        half_nprocs = torch.get_num_threads()
        torch.set_num_threads((half_nprocs * 2)//world_size)
        dist.init_process_group(
            backend='gloo', init_method=args.dist_url,
            world_size=world_size, rank=rank)

    if device != 'cpu':
        torch.cuda.set_device(device)

    model = gen_model(args, device)

    testset = ScpDataset(args.input_scp)
    data_sampler = InferenceDistributedSampler(testset)
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, sampler=data_sampler, collate_fn=TestPadCollate())

    output = []
    L = sum([1 for _ in testloader])
    with autocast(enabled=(True if device != 'cpu' else False)):
        for i, batch in enumerate(testloader):
            key, x, x_lens = batch
            x = x.to(device)
            encoder_o, _ = model.encoder(x, x_lens)
            output.append((key[0], encoder_o.cpu()))
            print(
                "\r|{:<60}|[{:>5}/{:<5}]".format(int((i+1)/L*60)*'#', i+1, L), end='')
    print("\r|{0:<60}|[{1:>5}/{1:<5}]".format(60*'#', L))
    with open(f'{suffix}.{gpu}', 'wb') as fo:
        pickle.dump(output, fo)


def load_checkpoint(model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel], path_ckpt: str) -> torch.nn.Module:

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
    parser.add_argument("--lm-weight", type=float, default=0.1,
                        help="Weight of external LM.")

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--enc-out-dir", type=str, default=None)
    parser.add_argument("--mode", type=str,
                        choices=['greedy', 'beam'], default='beam')
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--spmodel", type=str, default='',
                        help="SPM model location.")
    parser.add_argument("--nj", type=int, default=None)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--lower", action='store_true', default=False)

    args = parser.parse_args()

    main(args)
