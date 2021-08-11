"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

import utils
import os
import argparse
import numpy as np
import model as model_zoo
import dataset as DataSet
from _specaug import SpecAug
from _layers import LSTMPredictNet
from collections import OrderedDict
from typing import Union, Tuple, Sequence
from warp_rnnt import rnnt_loss as RNNTLoss

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import ctc_crf_base


def main(args):
    if not torch.cuda.is_available():
        utils.highlight_msg("CPU only training is unsupported.")
        return None

    os.makedirs(args.dir+'/ckpt', exist_ok=True)
    setattr(args, 'iscrf', False)
    setattr(args, 'ckptpath', args.dir+'/ckpt')
    if os.listdir(args.ckptpath) != [] and not args.debug and args.resume is None:
        raise FileExistsError(
            f"{args.ckptpath} is not empty! Refuse to run the experiment.")

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"Global number of GPUs: {args.world_size}")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.batch_size = args.batch_size // ngpus_per_node

    if args.h5py:
        data_format = "hdf5"
        utils.highlight_msg("H5py reading might cause error with Multi-GPUs.")
        Dataset = DataSet.SpeechDataset
        if args.trset is None or args.devset is None:
            raise FileNotFoundError(
                "With '--hdf5' option, you must specify data location with '--trset' and '--devset'.")
    else:
        data_format = "pickle"
        Dataset = DataSet.SpeechDatasetPickle

    if args.trset is None:
        args.trset = os.path.join(args.data, f'{data_format}/tr.{data_format}')
    if args.devset is None:
        args.devset = os.path.join(
            args.data, f'{data_format}/cv.{data_format}')

    tr_set = Dataset(args.trset)
    test_set = Dataset(args.devset)

    train_sampler = DistributedSampler(tr_set)
    test_sampler = DistributedSampler(test_set)
    test_sampler.set_epoch(1)

    trainloader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, collate_fn=DataSet.sortedPadCollateTransducer())

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=DataSet.sortedPadCollateTransducer())

    logger = OrderedDict({
        'log_train': ['epoch,loss,loss_real,net_lr,time'],
        'log_eval': ['loss_real,time']
    })
    manager = utils.Manager(logger, build_model, args)

    # get GPU info
    gpu_info = utils.gather_all_gpu_info(args.gpu)

    if args.rank == 0:
        print("> Model built.")
        print("Model size:{:.2f}M".format(
            utils.count_parameters(manager.model)/1e6))

        utils.gen_readme(args.dir+'/readme.md',
                         model=manager.model, gpu_info=gpu_info)

    
    ########## DEBUG CODE ###########
    
    if args.decode:
        for i, minibatch in enumerate(testloader):
            logits, input_lengths, labels, label_lengths, path_weights = minibatch
            logits, labels, input_lengths, label_lengths = logits.cuda(
                args.gpu, non_blocking=True), labels, input_lengths, label_lengths

            predict = manager.model.module.decode(logits, input_lengths)
            print(predict)
            print(labels)
            break
        return
    #################################

    # training
    manager.run(train_sampler, trainloader, testloader, args)


class Transducer(nn.Module):
    def __init__(self, encoder: nn.Module = None, decoder: nn.Module = None, jointnet: nn.Module = None, specaug: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = jointnet
        self.specaug = specaug

    @torch.no_grad()
    def decode(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor, mode='greedy') -> torch.LongTensor:
        encoder_output, o_lens = self.encoder(inputs, input_lengths)
        bos_token = inputs.new_zeros(1, 1).to(torch.long)
        decoder_output_init, hidden_dec_init = self.decoder(bos_token)
        outputs = []
        for seq, T in zip(encoder_output, o_lens):
            pred_tokens = []
            decoder_output = decoder_output_init
            hidden_dec = hidden_dec_init
            for t in range(T):
                step_out = self.joint(seq[t], decoder_output.view(-1))
                
                ########## DEBUG CODE ###########
                # print(step_out)
                #################################
                
                pred = step_out.argmax(dim=0)
                pred = int(pred.item())
                if pred != 0:
                    pred_tokens.append(pred)
                decoder_input = torch.tensor([[pred]], device=inputs.device, dtype=torch.long)
                decoder_output, hidden_dec = self.decoder(
                    decoder_input, hidden_dec)

            outputs.append(torch.tensor(pred_tokens, device=inputs.device, dtype=torch.long))

        return utils.pad_list(outputs).to(torch.long)

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor, input_lengths: torch.LongTensor, target_lengths: torch.LongTensor) -> torch.FloatTensor:

        output_encoder, o_lens = self.encoder(inputs, input_lengths)
        padded_targets = torch.cat(
            [targets.new_zeros((targets.size(0), 1)), targets], dim=-1)
        output_decoder, _ = self.decoder(padded_targets)

        ########## DEBUG CODE ###########
        # print("Encoder output:", output_encoder.size())
        # print("Decoder output:", output_decoder.size())
        #################################

        joint_out = self.joint(output_encoder, output_decoder)

        ########## DEBUG CODE ###########
        # print("JointNet output:", joint_out.size())
        #################################

        loss = RNNTLoss(joint_out, targets.to(dtype=torch.int32), o_lens.to(
            dtype=torch.int32), target_lengths.to(dtype=torch.int32), reduction='mean', gather=True)

        ########## DEBUG CODE ###########
        # print(loss)
        # exit(0)
        #################################

        return loss


class JointNet(nn.Module):
    """
    Joint `encoder_output` and `decoder_output`.
    Args:
        encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size ``(batch, time_steps, dimensionA)``
        decoder_output (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size ``(batch, label_length, dimensionB)``
    Returns:
        outputs (torch.FloatTensor): outputs of joint `encoder_output` and `decoder_output`. `FloatTensor` of size ``(batch, time_steps, label_length, dimensionA + dimensionB)``
    """

    def __init__(self, odim_encoder: int, odim_decoder: int, num_classes: int):
        super().__init__()
        in_features = odim_encoder+odim_decoder
        self.fc_enc = nn.Linear(odim_encoder, in_features)
        self.fc_dec = nn.Linear(odim_decoder, in_features)
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, encoder_output: torch.FloatTensor, decoder_output: torch.FloatTensor) -> torch.FloatTensor:
        assert (encoder_output.dim() == 3 and decoder_output.dim() == 3) or (encoder_output.dim() == 1 and decoder_output.dim() == 1)

        
        ########## DEBUG CODE ###########
        # print("encoder output to join:", encoder_output.size())
        # print("decoder output to join:", decoder_output.size())
        #################################
        
        encoder_output = self.fc_enc(encoder_output)
        decoder_output = self.fc_dec(decoder_output)

        if encoder_output.dim() == 3:
            # expand the outputs
            input_length = encoder_output.size(1)
            target_length = decoder_output.size(1)

            encoder_output = encoder_output.unsqueeze(2)
            decoder_output = decoder_output.unsqueeze(1)

            encoder_output = encoder_output.repeat(
                [1, 1, target_length, 1])
            decoder_output = decoder_output.repeat(
                [1, input_length, 1, 1])

        # concat_outputs = torch.cat([encoder_output, decoder_output], dim=-1)
        # outputs = self.fc(concat_outputs).log_softmax(dim=-1)
        outputs = self.fc(encoder_output+decoder_output).log_softmax(dim=-1)

        return outputs


def build_model(args, configuration, train=True) -> nn.Module:
    def _build_encoder(config) -> Tuple[nn.Module, Union[nn.Module, None]]:
        NetKwargs = config['kwargs']
        _encoder = getattr(model_zoo, config['type'])(**NetKwargs)

        # FIXME: this is a hack
        _encoder.classifier = nn.Identity()

        if 'specaug_config' not in config:
            specaug = None
            if args.rank == 0:
                utils.highlight_msg("Disable SpecAug.")
        else:
            specaug = SpecAug(**config['specaug_config'])
        return _encoder, specaug

    def _build_decoder(config) -> nn.Module:
        NetKwargs = config['kwargs']
        # FIXME: flexible decoder network like encoder.
        return LSTMPredictNet(**NetKwargs)

    def _build_jointnet(config) -> nn.Module:
        """
            The joint network accept the concatence of outputs of the 
            encoder and decoder. So the input dimensions MUST match that.
        """
        NetKwargs = config['kwargs']
        return JointNet(**NetKwargs)

    assert 'encoder' in configuration and 'decoder' in configuration and 'joint' in configuration

    encoder, specaug = _build_encoder(configuration['encoder'])
    decoder = _build_decoder(configuration['decoder'])
    jointnet = _build_jointnet(configuration['joint'])

    model = Transducer(encoder=encoder, decoder=decoder,
                       jointnet=jointnet, specaug=specaug)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recognition argument")

    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Distributed Data Parallel')

    parser.add_argument("--seed", type=int, default=0,
                        help="Manual seed.")

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument("--decode", action="store_true",
                        help="Configure to debug settings, would overwrite most of the options.")

    parser.add_argument("--debug", action="store_true",
                        help="Configure to debug settings, would overwrite most of the options.")
    parser.add_argument("--h5py", action="store_true",
                        help="Load data with H5py, defaultly use pickle (recommended).")

    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of training procedure.")

    parser.add_argument("--data", type=str, default=None,
                        help="Location of training/testing data.")
    parser.add_argument("--trset", type=str, default=None,
                        help="Location of training data. Default: <data>/[pickle|hdf5]/tr.[pickle|hdf5]")
    parser.add_argument("--devset", type=str, default=None,
                        help="Location of dev data. Default: <data>/[pickle|hdf5]/cv.[pickle|hdf5]")
    parser.add_argument("--dir", type=str, default=None, metavar='PATH',
                        help="Directory to save the log and model files.")

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:13943', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    if args.debug:
        utils.highlight_msg("Debugging.")

    main(args)
