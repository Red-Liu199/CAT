#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# https://github.com/pytorch/fairseq/blob/main/scripts/average_checkpoints.py
"""
usage:
    python utils/avgmodel.py -h
"""

from cat.shared.manager import F_CHECKLIST, CheckpointManager

import os
import argparse
import collections
from typing import Literal, List

import torch


def average_checkpoints(inputs: List[str]):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        if not os.path.isfile(fpath):
            raise RuntimeError(f"{fpath} is not a checkpoint file.")

        state = torch.load(fpath, map_location='cpu')

        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(fpath, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] = torch.div(
                averaged_params[k], num_models, rounding_mode='floor')
    new_state["model"] = averaged_params
    return new_state


def select_checkpoint(fdir: str, n: int = -1, mode: Literal['best', 'last', 'slicing'] = 'best', _slicing: tuple = None):

    if os.path.isfile(fdir):
        cm = CheckpointManager(fdir)
    elif os.path.isdir(fdir):
        cm = CheckpointManager(os.path.join(fdir, F_CHECKLIST))
    else:
        raise RuntimeError(f"'{fdir}' is neither a file or a folder.")

    checklist = [(path, check['metric']) for path, check in cm.content.items()]

    if mode == 'best':
        assert n > 0
        assert n <= len(checklist)
        checklist = sorted(checklist, key=lambda x: x[1])[:n]
    elif mode == 'last':
        assert n > 0
        assert n <= len(checklist)
        checklist = checklist[-n:]
    else:
        assert _slicing is not None
        checklist = checklist[_slicing[0]:_slicing[1]]
    return [path for path, _ in checklist]


def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', type=str, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    parser.add_argument("--num-best", type=int, help='If set, try to find checkpoint.xxx.pt with N best metric')

    parser.add_argument("--num-checkpoints", type=int, help='If set, try to find checkpoint.xxx.pt int the path specified by input'
    "Set number of consecutive checkpoints to be average.")
    parser.add_argument("--upper-bound", type=int, help="Set upper bound checkpoint. E.g."
    '--num-checkpoint=10 --upper-bound=50, checkpoints 41-50 would be averaged.')
    # fmt: on
    args = parser.parse_args()

    if args.num_best is not None:
        assert os.path.isdir(args.inputs[0])
        list_path = select_checkpoint(args.inputs[0],  args.num_best)
        if args.output is None:
            args.output = os.path.join(
                args.inputs[0], f'avg_best_{args.num_best}.pt')
    else:
        num = None
        if args.num_checkpoints is not None:
            num = args.num_checkpoints

        assert not ((args.upper_bound is None) ^ (args.num_checkpoints is None)
                    ), "--upper-bound and --num-checkpoints are required together"

        list_path = []
        if num is None:
            list_path = args.inputs
            assert args.output is not None
        else:
            upper = args.upper_bound
            fdir = args.inputs[0]
            assert os.path.isdir(fdir), f'--inputs={fdir} is not a directory.'
            list_path = select_checkpoint(
                fdir, num, mode='slicing', _slicing=(upper-num+1, upper+1))

            if args.output is None:
                args.output = os.path.join(
                    fdir, f'avg_{upper}_{num}.pt')

    print("Averaging checkpoints:\n"+'\n'.join(list_path))
    new_state = average_checkpoints(list_path)
    torch.save(new_state, args.output)
    print("Finished writing averaged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
