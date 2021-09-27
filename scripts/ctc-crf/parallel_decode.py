"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

Parallel decode with multi-gpu and single-gpu-multi-process support 
"""

import coreutils
from lm_train import build_model as lm_build
from dataset import ScpDataset, TestPadCollate, FeatureReader
from transducer_train import build_model  # , Transducer, ConvJointNet
# from beam_search_base import BeamSearchRNNTransducer, BeamSearchConvTransducer, ConvMemBuffer
from beam_search_transducer import TransducerBeamSearcher
from beam_search_espnet import BeamSearchTransducer

import os
import json
import pickle
import argparse
import sentencepiece as spm
from tqdm import tqdm
import kaldiio
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
        args.cpu = True
    # L_set = sum(1 for _ in open(args.input_scp, 'r'))
    # indices = equalSplitIdx(L_set, world_size)
    if args.cpu:
        world_size = args.nj
    else:
        world_size = torch.cuda.device_count() * args.nj

    indices, sorted_scp = equalLenSplit(args.input_scp, world_size)
    args.input_scp = sorted_scp

    binary_enc = os.path.join(
        args.enc_out_dir, f"{os.path.basename(args.input_scp)}.enc.hidB")
    link_enc = os.path.join(
        args.enc_out_dir, f"{os.path.basename(args.input_scp)}.enc.hidL")
    if not os.path.isfile(binary_enc) or not os.path.isfile(link_enc):
        print("> Encoder output file not found, generating...")
        gen_encode_hidden(args, enc_bin=binary_enc, enc_link=link_enc)
    setattr(args, 'enc_hid_bin', binary_enc)
    setattr(args, 'enc_hid_link', link_enc)

    if args.cpu:
        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, indices))
        return None

    if world_size == 1:
        single_worker(args, device=0)
        return None

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
    sorted_scp = f"{scp_in}.sorted"
    linfo = f"{scp_in}.lens"
    if not os.path.isfile(sorted_scp) or not os.path.isfile(linfo):
        print("> Generate sorted dataset, might take a while...")
        dataset = []
        freader = FeatureReader()
        with open(scp_in, 'r') as fi:
            for line in fi:
                key, m_path = line.split()
                mat = freader(m_path)
                dataset.append([key, m_path, mat.shape[0]])
        dataset = sorted(dataset, key=lambda item: item[2], reverse=True)
        with open(sorted_scp, 'w') as fo:
            for key, m_path, _ in dataset:
                fo.write(f"{key} {m_path}\n")

        with open(linfo, 'wb') as fo:
            pickle.dump([L for _, _, L in dataset], fo)

    with open(linfo, 'rb') as fi:
        linfo = pickle.load(fi)

    linfo = [x**1.2 for x in linfo]
    if idx_end == -1:
        idx_end = len(linfo)

    L = idx_end - idx_beg
    if L < N:
        raise RuntimeError(f"len(set) < N: {L} < {N}")

    # greedy not optimal
    avg = sum(linfo)/N
    indices = [0]
    cnt_interval = 0
    cnt_parts = 0
    for i, l in enumerate(linfo):
        cnt_interval += l
        if cnt_interval >= avg:
            indices.append(i+1)
            cnt_parts += 1
            cnt_interval = 0
            if cnt_parts < N:
                avg = sum(linfo[indices[-1]:])/(N-cnt_parts)

    assert len(indices) == N+1

    return indices, sorted_scp


def main_worker(rank: int, args: argparse.Namespace, intervals: List[int]):

    gpu = rank // args.nj
    if args.cpu:
        device = 'cpu'
        half_nprocs = torch.get_num_threads()
        torch.set_num_threads((half_nprocs * 2)//(len(intervals)-1))
    else:
        device = gpu
    single_worker(
        args, device, idx_beg=intervals[rank], idx_end=intervals[rank+1], suffix='{}-{}'.format(gpu, rank))
    return None


def single_worker(args: argparse.Namespace, device: Union[int, str], idx_beg: int = 0, idx_end: int = -1, suffix: str = '0-0'):

    if device != 'cpu':
        torch.cuda.set_device(device)

    if args.lm_weight == 0.0 or args.lm_dir is None:
        model = gen_model(args, device)
        ext_lm = None
    else:
        model, ext_lm = gen_model(args, device, use_ext_lm=True)

    testset = ScpDataset(args.input_scp, idx_beg=idx_beg, idx_end=idx_end)
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=TestPadCollate())

    writer = os.path.join(args.output_dir, f'decode.{suffix}.tmp')
    if args.mode == 'beam':
        # if isinstance(model.joint, ConvJointNet):
        #     beamsearcher = BeamSearchConvTransducer(
        #         model, kernel_size=(3, 3), beam_size=args.beam_size)
        # else:
        #     beamsearcher = BeamSearchRNNTransducer(
        #         model, beam_size=args.beam_size)
        # beamsearcher = beamsearcher.to(device)
        # beamsearcher = BeamSearchTransducer(model.decoder, model.joint, args.beam_size,
        # lm=ext_lm, lm_weight=args.lm_weight)
        beamsearcher = TransducerBeamSearcher(model.decoder, model.joint, 0, args.beam_size,
                                              state_beam=2.3, expand_beam=2.3, lm_module=ext_lm, lm_weight=args.lm_weight)
        del model
    else:
        beamsearcher = None
    decode(args, beamsearcher, testloader,
           device=device, local_writer=writer)


@torch.no_grad()
def decode(args, beamsearcher, testloader, device, local_writer):
    f_enc_hid = open(args.enc_hid_bin, 'rb')
    with open(args.enc_hid_link, 'rb') as fi:
        f_enc_seeks = pickle.load(fi)

    def _load_enc_mat(k: str):
        f_enc_hid.seek(f_enc_seeks[k])
        return pickle.load(f_enc_hid)

    sp = spm.SentencePieceProcessor(model_file=args.spmodel)
    results = []

    L = len(testloader)
    for i, batch in enumerate(testloader):
        key, _, _ = batch
        enc_o = _load_enc_mat(key[0])
        enc_o = enc_o.to(device)
        pred = beamsearcher(enc_o)

        if isinstance(pred, tuple):
            pred = pred[0]

        if isinstance(beamsearcher, BeamSearchTransducer):
            pred = pred[0].yseq

        seq = sp.decode(pred)
        results.append((key, seq))
        print(
            "\r|{:<60}|[{:>5}/{:<5}]".format(int((i+1)/L*60)*'#', i+1, L), end='')
    print("\r|{0}|[{1:>5}/{1:<5}]".format(60*'#', L))
    with open(local_writer, 'w') as fi:
        for key, pred in results:
            assert len(key) == 1
            fi.write("{} {}\n".format(key[0], pred[0]))

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


@torch.no_grad()
def gen_encode_hidden(args, enc_bin: str, enc_link: str):
    if torch.cuda.is_available() and not args.cpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = gen_model(args, device)

    testset = ScpDataset(args.input_scp)
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, collate_fn=TestPadCollate())

    fseeks = {}
    with open(enc_bin, 'wb') as fo:
        L = len(testloader)
        for i, batch in enumerate(testloader):
            key, x, x_lens = batch
            x = x.to(device)

            encoder_o, _ = model.encoder(x, x_lens)
            fseeks[key[0]] = fo.tell()
            pickle.dump(encoder_o.cpu(), fo)
            print(
                "\r|{:<60}|[{:>5}/{:<5}]".format(int((i+1)/L*60)*'#', i+1, L), end='')
    print("")
    with open(enc_link, 'wb') as fo:
        pickle.dump(fseeks, fo)

    del model, x, x_lens, encoder_o, fseeks, testset, testloader
    torch.cuda.empty_cache()


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
    parser.add_argument("--lm-weight", type=float, default=1.0,
                        help="Weight of external LM.")

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--enc-out-dir", type=str, default=None)
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
