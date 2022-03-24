"""
Search weights factor for given N-best list files.
Currently only support searching two params
"""

import os
import uuid
import shutil
import argparse
from typing import Union, Dict

from interpolate_nbests import GetParser as InterpolateParser
from interpolate_nbests import main as interpolate_main
from asr_process import updateNamespaceFromDict

from wer import WERParser
from wer import main as WERMain


def main(args: argparse):
    assert args.ground_truth is not None and os.path.isfile(args.ground_truth)
    assert len(args.nbestlist) == len(args.search), (
        "\n"
        f"--nbestlist {args.nbestlist}\n"
        f"--search    {args.search}")
    assert all([(x == 0) or (x == 1) for x in args.search])
    num_param_search = sum(args.search)
    assert num_param_search >= 1, f"you should at least specify one parameter for searching, instead: {args.search}"

    params = {}
    if len(args.range) == 2:
        params.update({'range': [args.range for _ in range(num_param_search)]})
    elif len(args.range) / num_param_search == 2:
        params.update({'range': [args.range[2*i:2*i+2]
                      for i in range(num_param_search)]})
    else:
        raise ValueError(
            "invalid --range, expected one of these:\n"
            "1. '--range a b' setup range of all searching params to [a, b]\n"
            "2. '--range a_0 b_0 a_1 b_1 ... a_N b_N', setup range [a_0, b_0] for param0, ... [a_N, b_N] for param N.\n"
            f"However, given {num_param_search} params to search, your input is: '--range {' '.join(str(x) for x in args.range)}'")

    num_param_fixed = len(args.nbestlist) - num_param_search
    if num_param_fixed > 0:
        # generate temp nbest list for fixed-param files.
        cache_file = os.path.join('/tmp', str(uuid.uuid4())+'.nbest')
        f_nbest_fixed = [f_nbest for i, f_nbest in enumerate(
            args.nbestlist) if args.search[i] == 0]

        if args.weight is None:
            weight_fixed = [1.0]*num_param_fixed
        else:
            weight_fixed = args.weight
            assert len(weight_fixed) == num_param_fixed, (
                "\n"
                f"--weight {weight_fixed}\n"
                f"--search {args.search}")

        if len(f_nbest_fixed) == 1 and weight_fixed[0] == 1.0:
            shutil.copyfile(f_nbest_fixed[0], cache_file)
        else:
            interpolate_main(updateNamespaceFromDict(
                {'nbestlist': f_nbest_fixed,
                 'weights': weight_fixed,
                 }, InterpolateParser(), [cache_file]
            ))
        variable_list = [f_nbest for i, f_nbest in enumerate(
            args.nbestlist) if args.search[i] == 1]
        tuned_list = [cache_file] + variable_list
    else:
        tuned_list = args.nbestlist

    def evaluate(tuned_metric, _searchout):
        mapkey = ':'.join([str(x) for x in tuned_metric])
        if mapkey in _searchout:
            return _searchout[mapkey]['wer']
        cache_file = os.path.join('/tmp', str(uuid.uuid4())+'.nbest')
        interpolate_main(updateNamespaceFromDict(
            {
                'nbestlist': tuned_list,
                'weights': tuned_metric if num_param_fixed == 0 else ([1.0] + tuned_metric),
                'one-best': True
            }, InterpolateParser(), [cache_file]
        ))
        print('tunning: ' +
              ' | '.join([f"{x:4.2f}" for x in tuned_metric]) + '  ', end='')
        wer = WERMain(updateNamespaceFromDict(
            {
                'stripid': True,
                'cer': args.cer,
                'force-cased': args.force_cased
            }, WERParser(), [args.ground_truth, cache_file]
        ))
        os.remove(cache_file)
        _searchout[mapkey] = wer
        return wer['wer']

    def update_tuned_metric(_searchout: dict):
        # e.g. tuned_metric = 0.5:-0.3
        _metrics = min(_searchout.keys(),
                       key=lambda k: _searchout[k]['wer'])  # type: str
        return [float(x) for x in _metrics.split(':')]

    if isinstance(args.interval, float) or len(args.interval) == 1:
        args.interval = args.interval if isinstance(
            args.interval, float) else args.interval[0]
        params.update(
            {'interval': [args.interval for _ in range(num_param_search)]})
    elif len(args.range) == num_param_search:
        params.update({'interval': args.interval})
    else:
        raise ValueError(
            "invalid --interval, expected one of these:\n"
            "1. '--interval i' setup interval of all searching params to i\n"
            "2. '--interval i_0 i_1 ... i_N', setup interval i_0 for param0, ... i_N for param N.\n"
            f"However, given {num_param_search} params to search, your input is: '--interval {' '.join(str(x) for x in args.range)}'")

    n_iter = 1
    tuned_metric = [sum(params['range'][idx]) /
                    2 for idx in range(num_param_search)]
    last_metric = [None for _ in range(num_param_search)]
    searchout = {}  # type: Dict[str, Dict[str, Union[float, int]]]
    while last_metric != tuned_metric:
        print(
            f"Iter: {n_iter} | {' | '.join([str(x) for x in params['range']])}")
        last_metric = tuned_metric.copy()
        for idx_param in range(num_param_search):
            lower, upper = params['range'][idx_param]
            boundary_perf = [None, None]
            while upper - lower >= params["interval"][idx_param]:
                if boundary_perf[0] is None:
                    tuned_metric[idx_param] = lower
                    boundary_perf[0] = evaluate(tuned_metric, searchout)

                if boundary_perf[1] is None:
                    tuned_metric[idx_param] = upper
                    boundary_perf[1] = evaluate(tuned_metric, searchout)

                # update lower, upper for next loop
                if boundary_perf[0] > boundary_perf[1]:
                    boundary_perf[0] = None
                    lower = (lower+upper)/2
                else:
                    boundary_perf[1] = None
                    upper = (lower+upper)/2
            tuned_metric = update_tuned_metric(searchout)

            # if hits the boundary, extend range
            if tuned_metric[idx_param] in params['range'][idx_param]:
                _radius = (params['range'][idx_param][1] -
                           params['range'][idx_param][0]) / 2
                params['range'][idx_param] = [tuned_metric[idx_param] -
                                              _radius, tuned_metric[idx_param]+_radius]

        n_iter += 1

    if num_param_fixed > 0:
        os.remove(tuned_list[0])
    del evaluate
    del update_tuned_metric

    best_wer = min(searchout.values(), key=lambda item: item['wer'])
    print(f"{best_wer['string']}\t{tuned_metric}")
    return tuned_metric, min(searchout.values(), key=lambda item: item['wer'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbestlist", type=str, nargs='+',
                        help="N-best list files")
    parser.add_argument("--search", type=int, nargs='+', choices=[0, 1], default=[0],
                        help="Flag of whether search weight of the file or not. ")
    parser.add_argument("--weight", type=float, nargs='*',
                        help="Weights of fixed parts, # which should be the same as # '0' in --search. defaults are all 1.0.")
    parser.add_argument("--range", type=float, nargs='+', default=[0., 1.],
                        help="Range of parameter searching.")
    parser.add_argument("--interval", type=float, nargs='+', default=0.1,
                        help="Minimal interval of parameter searching.")
    parser.add_argument("--ground-truth", type=str,
                        help="WER.py: Ground truth text file.")
    parser.add_argument("--cer", action="store_true", default=False,
                        help="WER.py: Compute CER instead WER. Default: False")
    parser.add_argument("--force-cased", action="store_true",
                        help="WER.py: Force text to be the same cased.")

    args = parser.parse_args()

    main(args)
