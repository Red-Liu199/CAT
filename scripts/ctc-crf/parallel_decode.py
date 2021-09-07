import os
import json
import coreutils
import argparse
import sentencepiece as spm
from tqdm import tqdm
from transducer_train import build_model, Transducer, ConvJointNet
from dataset import ScpDataset, TestPadCollate
from collections import OrderedDict
from typing import Union, List, Tuple

from torch.utils.data import DataLoader
import torch
import torch.multiprocessing as mp


def main(args):

    if not os.path.isfile(args.spmodel):
        raise FileNotFoundError(
            "Invalid sentencepiece model location: {}".format(args.spmodel))

    if not torch.cuda.is_available() or args.cpu:
        coreutils.highlight_msg("Using CPU")
        single_worker(args, 'cpu')
        return None

    world_size = torch.cuda.device_count()

    L_set = sum(1 for _ in open(args.input_scp, 'r'))
    intervals = L_set // world_size
    intervals = [intervals * i for i in range(world_size+1)]
    if intervals[-1] != L_set:
        intervals[-1] = L_set

    mp.spawn(main_worker, nprocs=world_size,
             args=(args, intervals))


def main_worker(gpu: int, args: argparse.Namespace, intervals: List[int]):

    num_processes = args.nj

    # NOTE: this is required for the ``fork`` method to work
    # model.share_memory()

    L = intervals[gpu+1] - intervals[gpu]
    _interval = L // num_processes
    sub_intervals = [_interval * i + intervals[gpu]
                     for i in range(num_processes+1)]

    if sub_intervals[-1] != intervals[gpu+1]:
        sub_intervals[-1] = intervals[gpu+1]

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=single_worker, args=(
            args, gpu, sub_intervals[rank], sub_intervals[rank+1], f'{gpu}-{rank}'))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def single_worker(args: argparse.Namespace, device: Union[int, str], idx_beg: int = 0, idx_end: int = -1, suffix: str = '0-0'):

    if device != 'cpu':
        torch.cuda.set_device(device)

    model = gen_model(args, device)
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
    results = []
    for batch in tqdm(testloader):
        # for batch in testloader:
        key, x, x_lens = batch
        x = x.to(device)

        if isinstance(model.joint, ConvJointNet):
            pred = model.decode_conv(
                x, x_lens, beam_size=args.beam_size, kernel_size=(3, 3))
        else:

            pred = model.decode(x, x_lens, mode=args.mode,
                                beam_size=args.beam_size)

        seq = sp.decode(pred.data.cpu().tolist())
        results.append((key, seq))

    with open(local_writer, 'w') as fi:
        for key, pred in results:
            assert len(key) == 1
            fi.write("{} {}\n".format(key[0], pred[0]))


def gen_model(args, device) -> torch.nn.Module:
    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures, dist=False, verbose=False)
    model = model.to(device)
    assert args.resume is not None, "Trying to decode with uninitialized parameters. Add --resume"
    if isinstance(device, int):
        device = f'cuda:{device}'
    model = load_checkpoint(model, args.resume, loc=device)
    return model


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


if __name__ == '__main__':

    parser = coreutils.BasicDDPParser(istraining=False)

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
