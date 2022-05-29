"""Ported from run_rnnt.sh, rewrote with python

Uage:
    python utils/pipeline/asr.py
"""

__all__ = [
    'checkExist',
    'combine_text',
    'dumpjson',
    'get_args',
    'initial_datainfo',
    'log_commit',
    'model_average',
    'mp_spawn',
    'readfromjson',
    'resolve_in_priority',
    'set_visible_gpus',
    'train_nn_model',
    'train_tokenizer'
]

from cat.shared import tokenizer as tknz
from cat.shared._constants import (
    F_DATAINFO,
    F_NN_CONFIG,
    F_HYPER_CONFIG,
    D_CHECKPOINT,
    D_INFER
)
from cat.utils.data import resolvedata


import os
import sys
import json
import uuid
import pickle
import argparse
from typing import *
from multiprocessing import Process


def initial_datainfo():
    if not os.path.isfile(F_DATAINFO):
        resolvedata.main()


def mp_spawn(target: Callable, args: Union[tuple, argparse.Namespace]):
    """Spawn a new process to execute the target function with given args."""
    if isinstance(args, argparse.Namespace):
        args = (args, )
    worker = Process(target=target, args=args)

    worker.start()
    worker.join()
    if worker.exitcode is not None and worker.exitcode != 0:
        sys.stderr.write("Worker unexpectedly terminated. See above info.\n")
        exit(1)


def readfromjson(file: str) -> dict:
    checkExist('f', file)
    with open(file, 'r') as fi:
        data = json.load(fi)
    return data


def dumpjson(obj: dict, target: str):
    assert os.access(os.path.dirname(target),
                     os.W_OK), f"{target} is not writable."
    with open(target, 'w') as fo:
        json.dump(obj, fo, indent=4)


def resolve_sp_path(config: dict, prefix: Optional[str] = None, allow_making: bool = False):
    if 'model_prefix' in config:
        spdir = os.path.dirname(config['model_prefix'])
        if not os.path.isdir(spdir):
            sys.stderr.write(
                f"WARNING: trying to resolve from an empty directory: {spdir}\n")
            if allow_making:
                os.makedirs(spdir)
        return config, (config['model_prefix']+'.model', config['model_prefix']+'.vocab')

    if prefix is not None:
        prefix += '_'
    else:
        prefix = ''

    assert 'model_type' in config, "resolve_sp_path: missing 'model_type' in configuration"
    if config['model_type'] == 'word' or config['model_type'] == 'char':
        f_out = prefix + config['model_type']
        config['use_all_vocab'] = True
    elif config['model_type'] == 'unigram':
        assert 'vocab_size' in config, "resolve_sp_path: missing 'vocab_size' in configuration"
        f_out = prefix + config['model_type'] + '_' + str(config['vocab_size'])
    else:
        raise NotImplementedError(
            f"Unknown tokenization mode: {config['model_type']}, expected one of ['word', 'char', 'unigram']")

    config['model_prefix'] = os.path.join('sentencepiece', f_out+'/spm')
    if allow_making:
        os.makedirs(os.path.dirname(config['model_prefix']), exist_ok=True)
    return config, (config['model_prefix']+'.model', config['model_prefix']+'.vocab')


def sp_train(intext: str, **kwargs):
    """Train the sentencepiece tokenizer.

    Args:
        intext   (str) : input text file for training. Each line represent a sentence.
        spdir    (str) : output directory to store sentencepiece model and vocab.
        **kwargs (dict): any keyword arguments would be parsed into sentencepiece training
    """
    try:
        import sentencepiece as spm
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Tokenization requires module sentencepiece. Install with\npip install sentencepiece")

    checkExist('f', intext)

    DEFAULT_SETTINGS = {
        "input": intext,
        "num_threads": os.cpu_count(),
        "bos_id": 0,
        "eos_id": -1,
        "unk_id": 1,
        "character_coverage": 1.0,
        "unk_surface": "<unk>"
    }
    DEFAULT_SETTINGS.update(kwargs)
    if 'user_defined_symbols' in DEFAULT_SETTINGS:
        if os.path.isfile(DEFAULT_SETTINGS['user_defined_symbols']):
            with open(DEFAULT_SETTINGS['user_defined_symbols'], 'r') as fi:
                syms = fi.readlines()
            DEFAULT_SETTINGS['user_defined_symbols'] = [x.strip()
                                                        for x in syms]

    # available options https://github.com/google/sentencepiece/blob/master/doc/options.md
    spm.SentencePieceTrainer.Train(**DEFAULT_SETTINGS)


def pack_data(
        f_scps: Union[List[str], str],
        f_labels: Union[List[str], str],
        f_out: str,
        filter: Optional[str] = None,
        tokenizer=None):
    """Parsing audio feature and text label into pickle file.

    Args:
        f_scps   (str, list): Kaldi-like-style .scp file(s).
        f_labels (str, list): Pure text file(s) include utterance id and sentence labels. Split by space.
        f_out   (str): Ouput pickle file location.
        filter (str, optional): identifier for filtering out seqs with unqualified length. 
            such as '100:2000' means remove those whose length is shorter than 100 or longer than 2000. Default: None
        tokenizer (AbsTokenizer, optional): If `tokenizer` is None, lines in `f_label` MUST be token indices, 
            otherwise it should be text.
    """
    import kaldiio
    import numpy as np
    from tqdm import tqdm

    if os.path.isfile(f_out):
        sys.stderr.write("warning: pack_data() "
                         f"file exists: {f_out}, "
                         "rm it if you want to update the data.\n\n")
        return

    if isinstance(f_scps, str):
        f_scps = [f_scps]
    if isinstance(f_labels, str):
        f_labels = [f_labels]

    checkExist('f', f_scps+f_labels)
    checkExist('d', os.path.dirname(f_out))

    l_min = 1
    l_max = float('inf')
    if filter is not None:
        assert ':' in filter, f"pack_data: invalid filter format {filter}"
        l_bound, u_bound = (i for i in filter.split(':'))
        if l_bound != '':
            l_min = int(l_bound)
        if u_bound != '':
            l_max = int(u_bound)

    # process label files
    labels = []
    for _f_lb in f_labels:
        with open(_f_lb, 'r') as fi_label:
            labels += fi_label.readlines()

    labels = [l.strip().split(maxsplit=1)
              for l in labels]      # type: List[Tuple[str, str]]
    num_label_lines = len(labels)

    if tokenizer is None:
        # assume the labels are given in number ids
        labels = {
            uid: np.asarray(
                [int(i) for i in utt.split()],
                dtype=np.int64
            )
            for uid, utt in labels
        }
    else:
        labels = {
            uid: np.asarray(
                tokenizer.encode(utt),
                dtype=np.int64
            )
            for uid, utt in labels
        }

    num_utts = [sum(1 for _ in open(_f_scp, 'r')) for _f_scp in f_scps]
    total_utts = sum(num_utts)
    assert total_utts == num_label_lines, \
        "pack_data: f_scp and f_label should match on the # lines, " \
        f"instead {total_utts} != {len(labels)}"

    f_opened = {}
    cnt_frames = 0
    linfo = np.empty(total_utts, dtype=np.int64)
    _keys = []
    _ark_seeks = []
    idx = 0
    cnt_rm = 0
    for n, _f_scp in enumerate(f_scps):
        with open(_f_scp, 'r') as fi_scp:
            for line in tqdm(fi_scp, total=num_utts[n]):
                key, loc_ark = line.split()
                mat = kaldiio.load_mat(
                    loc_ark, fd_dict=f_opened)   # type:np.ndarray

                if mat.shape[0] < l_min or mat.shape[0] > l_max:
                    cnt_rm += 1
                    continue

                linfo[idx] = mat.shape[0]
                _keys.append(key)
                _ark_seeks.append(loc_ark)

                cnt_frames += mat.shape[0]
                idx += 1

    for f in f_opened.values():
        f.close()

    # in order to store labels in a ndarray,
    # first I pad all labels to the max length with -1 (this won't take many memory since labels are short compared to frames)
    # then store the length in the last place, such as
    # [0 1 2 3] -> [0 1 2 3 -1 -1 4]
    # then we can access the data via array[:array[-1]]
    labels = list([labels[k_] for k_ in _keys])
    cnt_tokens = sum(_x.shape[0] for _x in labels)
    max_len_label = max(x.shape[0] for x in labels)
    labels = np.array([
        np.concatenate((
            _x,
            np.array([-1]*(max_len_label-_x.shape[0]) + [_x.shape[0]])
        ))
        for _x in labels])

    linfo = linfo[:idx]
    ark_locs = np.array(_ark_seeks)
    assert len(linfo) == len(labels)
    assert len(labels) == len(ark_locs)

    with open(f_out, 'wb') as fo:
        pickle.dump({
            'label': labels,
            'linfo': linfo,
            'arkname': ark_locs,
            'key': np.array(_keys)}, fo)

    if cnt_rm > 0:
        print(f"pack_data: remove {cnt_rm} unqualified sequences.")
    print(
        f"...# frames: {cnt_frames} | # tokens: {cnt_tokens} | # seqs: {idx}")


def checkExist(f_type: Literal['d', 'f'], f_list: Union[str, List[str]]):
    """Check whether directory/file exist and raise error if it doesn't.
    """
    assert f_type in [
        'd', 'f'], f"checkExist: Unknown f_type: {f_type}, expected one of ['d', 'f']"
    if isinstance(f_list, str):
        f_list = [f_list]
    assert len(
        f_list) > 0, f"checkExist: Expect the file/dir list to have at least one element, but found empty."

    hints = {'d': 'Directory', 'f': 'File'}
    not_founds = []

    if f_type == 'd':
        check = os.path.isdir
    elif f_type == 'f':
        check = os.path.isfile
    else:
        raise RuntimeError(
            f"checkExist: Unknown f_type: {f_type}, expected one of ['d', 'f']")

    for item in f_list:
        if not check(item):
            not_founds.append(item)

    if len(not_founds) > 0:
        o_str = f"{hints[f_type]} checking failed:"
        for item in not_founds:
            o_str += f"\n\t{item}"
        raise FileNotFoundError(o_str)
    else:
        return


def recursive_rpl(src_dict: dict, target_key: str, rpl_val):
    if not isinstance(src_dict, dict):
        return

    if target_key in src_dict:
        src_dict[target_key] = rpl_val
    else:
        for k, v in src_dict.items():
            recursive_rpl(v, target_key, rpl_val)


def get_args(src_dict: dict, parser: argparse.ArgumentParser, positionals: List = []):
    args = argparse.Namespace()
    processed_dict = {}
    for k, v in src_dict.items():
        processed_dict[k.replace('-', '_')] = v
    args.__dict__.update(processed_dict)
    to_args = parser.parse_args(args=positionals, namespace=args)
    return to_args


def set_visible_gpus(N: int) -> str:
    assert N >= 0
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(N))
    else:
        seen_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(seen_gpus[:N])
    return os.environ['CUDA_VISIBLE_DEVICES']


def get_free_port():
    '''Return a free available port on local machine.'''
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]


def train_nn_model(
        working_dir: str,
        f_hyper_p: str,
        fmt_data: str,
        promt: str = '{}\n'):

    checkExist('f', f_hyper_p)
    settings = readfromjson(f_hyper_p)
    assert 'train' in settings, promt.format("missing 'train' in field:")
    assert 'bin' in settings['train'], promt.format(
        "missing 'bin' in field:train:")
    assert 'option' in settings['train'], promt.format(
        "missing 'bin' in field:train:")

    if 'tokenizer' not in settings:
        sys.stderr.write(
            f"warning: missing 'tokenizer': {f_hyper_p}.\n"
            "... You have ensure the 'num_classes' in config is correct.\n")
    else:
        checkExist('f', settings['tokenizer']['location'])
        tokenizer = tknz.load(settings['tokenizer']['location'])

        f_nnconfig = os.path.join(working_dir, F_NN_CONFIG)
        checkExist('f', f_nnconfig)

        nnconfig = readfromjson(f_nnconfig)
        # recursively search for 'num_classes'
        recursive_rpl(nnconfig, 'num_classes', tokenizer.vocab_size)
        dumpjson(nnconfig, f_nnconfig)
        del tokenizer

    train_options = settings['train']['option']
    if 'trset' not in train_options:
        train_data = fmt_data.format('train')
        checkExist('f', train_data)
        train_options['trset'] = train_data
        sys.stdout.write(promt.format(f"set 'trset' to {train_data}"))
    if 'devset' not in train_options:
        dev_data = fmt_data.format('dev')
        checkExist('f', dev_data)
        train_options['devset'] = dev_data
        sys.stdout.write(promt.format(f"set 'devset' to {dev_data}"))
    if 'dir' not in train_options:
        train_options['dir'] = working_dir
        sys.stdout.write(promt.format(f"set 'dir' to {working_dir}"))
    if 'dist-url' not in train_options:
        train_options['dist-url'] = f"tcp://localhost:{get_free_port()}"
        sys.stdout.write(promt.format(
            f"set 'dist-url' to {train_options['dist-url']}"))

    import importlib
    interface = importlib.import_module(settings['train']['bin'])

    mp_spawn(interface.main, get_args(
        train_options, interface._parser()))


def resolve_in_priority(dataset: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Resolve text file location for dataset.

    Args:
        dataset (str, list): dataset(s)

    Returns:
        (local_texts, outside_texts)
    """
    if isinstance(dataset, str):
        dataset = [dataset]

    datainfo = readfromjson(F_DATAINFO)

    local_text = []     # find in local path, assume NO uid before each utterance
    outside_text = []   # find in src data, assume uid before each utterance
    for _set in dataset:
        if os.path.isfile(_set):
            local_text.append(_set)
        elif _set in datainfo:
            outside_text.append(datainfo[_set]['trans'])
        else:
            raise FileNotFoundError(f"request dataset: '{_set}' not found.")
    return local_text, outside_text


def combine_text(datasets: Union[str, List[str]], f_out: Optional[str] = None) -> str:
    """Combine text files of dataset(s) and return the combined file."""
    text_noid, text_withid = resolve_in_priority(datasets)
    assert len(text_noid) > 0 or len(
        text_withid) > 0, f"combineText: dataset '{datasets}' seems empty."

    if f_out is None:
        f_out = os.path.join('/tmp', str(uuid.uuid4()))

    with open(f_out, 'w') as fo:
        for _text in text_noid:
            with open(_text, 'r') as fi:
                fo.write(fi.read())
        for _text in text_withid:
            with open(_text, 'r') as fi:
                for line in fi:
                    # rm the seq id in first column
                    _, utt = line.split(maxsplit=1)
                    fo.write(utt)
    return f_out


def train_tokenizer(f_hyper: str):
    checkExist('f', f_hyper)
    hyper_settings = readfromjson(f_hyper)

    assert 'tokenizer' in hyper_settings, f"missing 'tokenizer': {f_hyper}"
    if 'location' not in hyper_settings['tokenizer']:
        sys.stderr.write(
            f"missing 'location' in ['tokenizer']: {f_hyper}\n")
        sys.stderr.write(f"set ['tokenizer']='tokenizer.tknz'\n")
        hyper_settings['tokenizer']['location'] = os.path.join(
            os.path.dirname(f_hyper), 'tokenizer.tknz')
    else:
        if os.path.isfile(hyper_settings['tokenizer']['location']):
            sys.stderr.write(
                f"['tokenizer']['location'] exists: {hyper_settings['tokenizer']['location']}\n"
                "...skip tokenizer training. If you want to do tokenizer training anyway,\n"
                "...remove the ['tokenizer']['location'] in setting\n"
                f"...or remove the file:{hyper_settings['tokenizer']['location']} then re-run the script.\n")
            return

    assert 'type' in hyper_settings[
        'tokenizer'], f"missing property 'type' in ['tokenizer']: {f_hyper}"
    assert 'data' in hyper_settings, f"missing property 'data': {f_hyper}"
    assert 'train' in hyper_settings[
        'data'], f"missing property 'train' in ['data']: {f_hyper}"
    assert os.access(os.path.dirname(hyper_settings['tokenizer']['location']),
                     os.W_OK), f"['tokenizer']['location'] is not writable: '{hyper_settings['tokenizer']['location']}'"

    if 'lang' in hyper_settings['data']:
        # check if it's chinese-like languages
        if ('zh' == hyper_settings['data']['lang'].split('-')[0]):
            sys.stderr.write(
                "TrainTokenizer(): for Asian language, it's your duty to remove the segment spaces up to your requirement.\n")

    tokenizer_type = hyper_settings['tokenizer']['type']
    if 'property' not in hyper_settings['tokenizer']:
        hyper_settings['tokenizer']['property'] = {}

    if tokenizer_type == 'SentencePieceTokenizer':
        f_corpus_tmp = combine_text(hyper_settings['data']['train'])
        sp_settings, (f_tokenizer, _) = resolve_sp_path(
            hyper_settings['tokenizer']['property'], os.path.basename(os.getcwd()), allow_making=True)
        sp_train(f_corpus_tmp, **sp_settings)
        hyper_settings['tokenizer']['property'] = sp_settings
        tokenizer = tknz.SentencePieceTokenizer(spmodel=f_tokenizer)
        hyper_settings['tokenizer']['property']['vocab_size'] = tokenizer.vocab_size
        os.remove(f_corpus_tmp)
    elif tokenizer_type == 'JiebaComposePhoneTokenizer':
        tokenizer = tknz.JiebaComposePhoneTokenizer(
            **hyper_settings['tokenizer']['property'])
    elif tokenizer_type == 'JiebaTokenizer':
        # jieba tokenizer doesn't need training
        tokenizer = tknz.JiebaTokenizer(
            **hyper_settings['tokenizer']['property'])
    else:
        raise ValueError(f"Unknown type of tokenizer: {tokenizer_type}")

    tknz.save(tokenizer, hyper_settings['tokenizer']['location'])

    dumpjson(hyper_settings, f_hyper)


def model_average(
        setting: dict,
        checkdir: str,
        returnifexist: bool = False) -> Tuple[str, str]:
    """Do model averaging according to given setting, return the averaged model path."""

    assert 'mode' in setting, "missing 'mode' in ['avgmodel']"
    assert 'num' in setting, "missing 'num' in ['avgmodel']"
    avg_mode, avg_num = setting['mode'], setting['num']

    import torch
    from cat.utils.avgmodel import select_checkpoint, average_checkpoints

    suffix_avgmodel = f"{avg_mode}-{avg_num}"
    checkpoint = os.path.join(checkdir, suffix_avgmodel)
    tmp_check = checkpoint + ".pt"
    if os.path.isfile(tmp_check) and returnifexist:
        return tmp_check, suffix_avgmodel
    i = 1
    while os.path.isfile(tmp_check):
        tmp_check = checkpoint + f".{i}.pt"
        i += 1
    checkpoint = tmp_check

    if avg_mode in ['best', 'last']:
        params = average_checkpoints(
            select_checkpoint(checkdir, avg_num, avg_mode))
    else:
        raise NotImplementedError(
            f"Unknown model averaging mode: {avg_mode}, expected in ['best', 'last']")

    # delete the parameter of optimizer for saving disk.
    for k in list(params.keys()):
        if k != 'model':
            del params[k]
    torch.save(params, checkpoint)
    return checkpoint, suffix_avgmodel


def log_commit(f_hyper: str):
    import subprocess
    if subprocess.run('command -v git', shell=True, capture_output=True).returncode != 0:
        sys.stderr.write(
            "warning: git command not found. Skip logging commit.\n")
    else:
        process = subprocess.run(
            "git log -n 1 --pretty=format:\"%H\"", shell=True, check=True, stdout=subprocess.PIPE)

        orin_settings = readfromjson(f_hyper)
        orin_settings['commit'] = process.stdout.decode('utf-8')
        dumpjson(orin_settings, f_hyper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("expdir", type=str, help="Experiment directory.")
    parser.add_argument("--start_stage", dest='stage_beg',
                        type=int, default=1, help="Start stage of processing. Default: 1")
    parser.add_argument("--stop_stage", dest='stage_end',
                        type=int, default=-1, help="Stop stage of processing. Default: last stage.")
    parser.add_argument("--ngpu", type=int, default=-1,
                        help="Number of GPUs to be used.")
    parser.add_argument("--silent", action="store_true",
                        help="Disable detailed messages output.")

    args = parser.parse_args()
    s_beg = args.stage_beg
    s_end = args.stage_end
    if s_end == -1:
        s_end = float('inf')

    assert s_end >= 1, f"Invalid stop stage: {s_end}"
    assert s_beg >= 1 and s_beg <= s_end, f"Invalid start stage: {s_beg}"

    cwd = os.getcwd()
    working_dir = args.expdir
    checkExist('d', working_dir)
    f_hyper = os.path.join(working_dir, F_HYPER_CONFIG)
    checkExist('f', f_hyper)
    initial_datainfo()
    datainfo = readfromjson(F_DATAINFO)
    hyper_cfg = readfromjson(f_hyper)
    if "env" in hyper_cfg:
        for k, v in hyper_cfg["env"].items():
            os.environ[k] = v
    if 'commit' not in hyper_cfg:
        log_commit(f_hyper)

    if args.ngpu > -1:
        set_visible_gpus(args.ngpu)

    ############ Stage 1  Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 1 Tokenizer training"))
        fmt = "# Tokenizer trainin # {}\n" if not args.silent else ""
        hyper_cfg = readfromjson(f_hyper)
        if 'tokenizer' not in hyper_cfg:
            sys.stderr.write(
                "warning: missing 'tokenizer' in hyper-setting, skip tokenizer training.")
        else:
            train_tokenizer(f_hyper)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 2 Pickle data"))
        fmt = "# Pickle data # {}\n" if not args.silent else ""

        hyper_cfg = readfromjson(f_hyper)
        assert 'data' in hyper_cfg, f"missing 'data': {f_hyper}"

        if 'tokenizer' not in hyper_cfg:
            sys.stderr.write(
                f"warning: missing 'tokenizer', assume the ground truth text files as tokenized ones.")
            istokenized = True
            tokenizer = None
        else:
            # load tokenizer from file
            assert 'location' in hyper_cfg[
                'tokenizer'], f"missing 'location' in ['tokenizer']: {f_hyper}"

            f_tokenizer = hyper_cfg['tokenizer']['location']
            checkExist('f', f_tokenizer)
            tokenizer = tknz.load(f_tokenizer)
            istokenized = False

        data_settings = hyper_cfg['data']
        if 'filter' not in data_settings:
            data_settings['filter'] = None

        d_pkl = os.path.join(working_dir, 'pkl')
        os.makedirs(d_pkl, exist_ok=True)
        for dataset in ['train', 'dev', 'test']:
            if dataset not in data_settings:
                sys.stderr.write(
                    f"warning: missing '{dataset}' in ['data'], skip.\n")
                continue
            sys.stdout.write(fmt.format(f"packing {dataset} data..."))

            if dataset == 'train':
                filter = data_settings['filter']
            else:
                filter = None

            if isinstance(data_settings[dataset], str):
                data_settings[dataset] = [data_settings[dataset]]
            f_data = []
            for _set in data_settings[dataset]:
                if _set not in datainfo:
                    raise RuntimeError(
                        f"'{_set}' not found. you can configure it manually in {F_DATAINFO}")
                f_data.append(datainfo[_set])

            pack_data(
                [_data['scp'] for _data in f_data],
                [_data['trans'] for _data in f_data],
                f_out=os.path.join(d_pkl, dataset+'.pkl'),
                filter=filter, tokenizer=tokenizer)
            del f_data

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 3 NN training"))
        fmt = "# NN training # {}\n" if not args.silent else ""

        train_nn_model(
            working_dir,
            f_hyper,
            f'{working_dir}'+'/pkl/{}.pkl',
            fmt
        )

    ############ Stage 4  Decode ############
    if s_beg <= 4 and s_end >= 4:
        # FIXME: runing script directly from NN training to decoding always producing SIGSEGV error
        if s_beg <= 3:
            os.system(" ".join([
                sys.executable,     # python interpreter
                sys.argv[0],        # file script
                working_dir,
                "--silent" if args.silent else "",
                "--start_stage=4",
                f"--stop_stage={args.stage_end}",
                f"--ngpu={args.ngpu}"
            ]))
            sys.exit(0)

        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 4 Decode"))
        fmt = "# Decode # {}\n" if not args.silent else ""

        hyper_cfg = readfromjson(f_hyper)
        assert 'inference' in hyper_cfg, f"missing 'inference' at {f_hyper}"

        cfg_infr = hyper_cfg['inference']

        checkdir = os.path.join(working_dir, D_CHECKPOINT)
        # do model averaging
        if 'avgmodel' in cfg_infr and os.path.isdir(checkdir):
            checkpoint = model_average(
                setting=cfg_infr['avgmodel'],
                checkdir=checkdir,
                returnifexist=True
            )[0]
        else:
            checkpoint = None

        # infer
        if 'infer' in cfg_infr:
            # try to get inference:infer:option
            assert 'bin' in cfg_infr['infer'], \
                f"missing 'bin' in inference:infer: at {f_hyper}"
            assert 'option' in cfg_infr['infer'], \
                f"missing 'option' in inference:infer: at {f_hyper}"

            infr_option = cfg_infr['infer']['option']
            # find checkpoint
            if infr_option.get('resume', None) is None:
                # no avgmodel found, get the best checkpoint
                if checkpoint is None and os.path.isdir(checkdir):
                    checkpoint = model_average(
                        setting={'mode': 'best', 'num': 1},
                        checkdir=checkdir,
                        returnifexist=True
                    )[0]
                # the last check, no fallback method, raise warning
                if checkpoint is None:
                    sys.stderr.write(
                        "warning: inference:infer:option:resume is none.\n"
                        "    which would causing non-initialized evaluation.\n"
                    )
                else:
                    # there's no way the output of model_average() is an invalid path
                    # ... so here we could skip the checkExist()
                    # update config to file
                    _hyper = readfromjson(f_hyper)
                    _hyper['inference']['infer']['option']['resume'] = checkpoint
                    dumpjson(_hyper, f_hyper)
                    infr_option['resume'] = checkpoint
                    sys.stdout.write(fmt.format(
                        f"set inference:infer:option:resume to {checkpoint}"))
            else:
                sys.stdout.write(fmt.format(
                    "setting 'resume' in inference:infer:option would ignore the inference:avgmodel settings."))
                checkpoint = infr_option['resume']
                checkExist('f', checkpoint)

            if 'config' not in infr_option:
                infr_option['config'] = os.path.join(working_dir, F_NN_CONFIG)
                checkExist('f', infr_option['config'])

            intfname = cfg_infr['infer']['bin']
            # check tokenizer
            if intfname != "cat.ctc.cal_logit":
                if 'tokenizer' not in infr_option:
                    assert hyper_cfg.get('tokenizer', {}).get('location', None) is not None, \
                        (
                        "\nyou should set at least one of:\n"
                        f"1. set tokenizer:location ;\n"
                        f"2. set inference:infer:option:tokenizer \n"
                    )
                    infr_option['tokenizer'] = hyper_cfg['tokenizer']['location']

            ignore_field_data = False
            os.makedirs(f"{working_dir}/decode", exist_ok=True)
            if intfname == 'cat.ctc.cal_logit':
                if 'input_scp' in infr_option:
                    ignore_field_data = True

                if 'output_dir' not in infr_option:
                    assert not ignore_field_data
                    infr_option['output_dir'] = os.path.join(
                        working_dir, D_INFER+'/{}')
                    sys.stdout.write(fmt.format(
                        f"set inference:infer:option:output_dir to {infr_option['output_dir']}"))
            elif intfname in ['cat.ctc.decode', 'cat.rnnt.decode']:
                if 'input_scp' in infr_option:
                    ignore_field_data = True
                if 'output_prefix' not in infr_option:
                    topo = intfname.split('.')[1]
                    assert not ignore_field_data, f"error: seem you forget to set 'output_prefix'"

                    # rm dirname and '.pt'
                    if checkpoint is None:
                        suffix_model = 'none'
                    else:
                        suffix_model = os.path.basename(
                            checkpoint).removesuffix('.pt')
                    prefix = f"{topo}_bs{infr_option.get('beam_size', 'dft')}_{suffix_model}"
                    # set output format
                    a = infr_option.get('alpha', 0)
                    b = infr_option.get('beta', 0)
                    if not (a == 0 and b == 0):
                        prefix += f'_elm-a{a}b{b}'
                    if topo == 'rnnt':
                        ilmw = infr_option.get('ilm_weight', 0)
                        if ilmw != 0:
                            prefix += f'_ilm{ilmw}'
                    infr_option['output_prefix'] = os.path.join(
                        working_dir,
                        f"{D_INFER}/{prefix}"+"_{}"
                    )
            else:
                ignore_field_data = True
                sys.stderr.write(
                    f"warning: interface '{intfname}' only support handcrafted execution.\n")

            import importlib
            interface = importlib.import_module(intfname)
            assert hasattr(interface, 'main'), \
                f"{intfname} module does not have method main()"
            assert hasattr(interface, '_parser'), \
                f"{intfname} module does not have method _parser()"
            if ignore_field_data:
                interface.main(get_args(
                    infr_option,
                    interface._parser()
                ))
            else:
                assert 'data' in hyper_cfg, f"missing 'data' at {f_hyper}"
                assert 'test' in hyper_cfg[
                    'data'], f"missing 'test' in data: at {f_hyper}"

                testsets = hyper_cfg['data']['test']
                if isinstance(testsets, str):
                    testsets = [testsets]
                f_scps = [datainfo[_set]['scp'] for _set in testsets]
                checkExist('f', f_scps)

                running_option = infr_option.copy()
                for _set, scp in zip(testsets, f_scps):
                    for k in infr_option:
                        if isinstance(infr_option[k], str) and '{}' in infr_option[k]:
                            running_option[k] = infr_option[k].format(_set)
                            sys.stdout.write(fmt.format(
                                f"{_set}: {k} -> {running_option[k]}"))
                    running_option['input_scp'] = scp
                    if intfname in ['cat.ctc.decode', 'cat.rnnt.decode']:
                        if os.path.isfile(running_option['output_prefix']):
                            sys.stdout.write(fmt.format(
                                f"{running_option['output_prefix']} exists, skip."))
                            continue

                    # FIXME: this canonot be spawned via mp_spawn, otherwise error would be raised
                    #        possibly due to the usage of mp.Queue
                    interface.main(get_args(
                        running_option,
                        interface._parser()
                    ))
        else:
            infr_option = {}

        # compute wer/cer
        if 'er' in cfg_infr:
            from cat.utils import wer as wercal
            err_option = cfg_infr['er']
            if 'hy' not in err_option:
                assert infr_option.get('output_prefix', None) is not None, \
                    "inference:er:hy is not set and cannot be resolved from inference:infer."

                err_option['hy'] = infr_option['output_prefix']

            if err_option.get('oracle', False):
                err_option['hy'] = err_option['hy'] + '.nbest'

            if '{}' in err_option['hy']:
                # input in format string
                testsets = hyper_cfg.get('data', {}).get('test', [])
                if isinstance(testsets, str):
                    testsets = [testsets]
                for _set in testsets:
                    sys.stdout.write(f"{_set}\t")
                    wercal.main(get_args(
                        err_option,
                        wercal._parser(),
                        [
                            datainfo[_set]['trans'],
                            err_option['hy'].format(_set)
                        ]
                    ))
                    sys.stdout.flush()
            else:
                assert 'gt' in err_option
                wercal.main(get_args(
                    err_option,
                    wercal._parser(),
                    [err_option['gt'], err_option['hy']]
                ))
