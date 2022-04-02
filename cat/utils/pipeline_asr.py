"""Ported from run_rnnt.sh, rewrote with python

Uage:
    python utils/pipeline_asr.py
"""

try:
    from resolvedata import main as resolve_srcdata
    from resolvedata import F_DATAINFO
except ModuleNotFoundError:
    print("seems you're trying to import module from pipeline_asr. ensure you know what you're doing.")

import os
import sys
import json
import uuid
import pickle
import argparse
from typing import Union, Literal, List, Tuple, Optional, Callable, Dict
from multiprocessing import Process


def initial_datainfo():
    if not os.path.isfile(F_DATAINFO):
        resolve_srcdata()


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


def sentencepiece_train(intext: str, **kwargs):
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


def parsingData(
        f_scps: Union[List[str], str],
        f_labels: Union[List[str], str],
        f_out: str,
        filter: Optional[str] = None,
        tokenizer=None,
        iszh: bool = False):
    """Parsing audio feature and text label into pickle file.

    Args:
        f_scps   (str, list): Kaldi-like-style .scp file(s).
        f_labels (str, list): Pure text file(s) include utterance id and sentence labels. Split by space.
        f_out   (str): Ouput pickle file location.
        filter (str, optional): identifier for filtering out seqs with unqualified length. 
            such as '100:2000' means remove those whose length is shorter than 100 or longer than 2000. Default: None
        tokenizer (AbsTokenizer, optional): If `tokenizer` is None, lines in `f_label` MUST be token indices, 
            otherwise it should be text.
        iszh (bool, optional): whether is chinese-liked lang (charater-based)
    """
    import kaldiio
    import numpy as np
    from tqdm import tqdm

    if os.path.isfile(f_out):
        sys.stderr.write("warning: parsingData() "
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
        assert ':' in filter, f"parsingData: invalid filter format {filter}"
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
    labels = [l.split() for l in labels]
    num_label_lines = len(labels)

    if tokenizer is None:
        # assume the labels are given in number ids
        labels = {l[0]: np.asarray(
            [int(i) for i in l[1:]], dtype=np.int64) for l in labels}
    else:
        if iszh:
            labels = {l[0]: np.asarray(tokenizer.encode(''.join(l[1:])), dtype=np.int64)
                      for l in labels}
        else:
            labels = {l[0]: np.asarray(tokenizer.encode(' '.join(l[1:])), dtype=np.int64)
                      for l in labels}

    num_utts = [sum(1 for _ in open(_f_scp, 'r')) for _f_scp in f_scps]
    total_utts = sum(num_utts)
    assert total_utts == num_label_lines, \
        "parsingData: f_scp and f_label should match on the # lines, " \
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
        print(f"parsingData: remove {cnt_rm} unqualified sequences.")
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


def updateNamespaceFromDict(src_dict: dict, parser: argparse.ArgumentParser, positionals: List = []):
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


def TrainNNModel(
        args: argparse.Namespace,
        settings: dict,
        f_hyper_p: str,
        fmt_data: str,
        Parser: argparse.ArgumentParser,
        MainFunc: Callable[[argparse.Namespace], None],
        promt: str = '{}'):
    assert 'train' in settings, promt.format("missing 'train' in hyper-p")

    if 'tokenizer' not in settings:
        sys.stderr.write(
            f"warning: missing property 'tokenizer': {f_hyper_p}\n")
    else:
        try:
            import cat
        except ModuleNotFoundError:
            sys.path.append(cwd)
        from cat.shared import tokenizer as tknz
        checkExist('f', settings['tokenizer']['location'])
        tokenizer = tknz.load(settings['tokenizer']['location'])

        f_nnconfig = os.path.join(args.expdir, 'config.json')
        checkExist('f', f_nnconfig)

        nnconfig = readfromjson(f_nnconfig)
        # recursively search for 'num_classes'
        recursive_rpl(nnconfig, 'num_classes', tokenizer.vocab_size)
        dumpjson(nnconfig, f_nnconfig)
        del tokenizer

    import subprocess
    if subprocess.run('command -v git', shell=True, capture_output=True).returncode != 0:
        sys.stderr.write(
            "warning: git command not found. Skip saving commit.\n")
    else:
        process = subprocess.run(
            "git log -n 1 --pretty=format:\"%H\"", shell=True, check=True, stdout=subprocess.PIPE)

        orin_settings = readfromjson(f_hyper_p)
        orin_settings['commit'] = process.stdout.decode('utf-8')
        dumpjson(orin_settings, f_hyper_p)

    training_settings = settings['train']
    if 'trset' not in training_settings:
        train_data = fmt_data.format('train')
        checkExist('f', train_data)
        training_settings['trset'] = train_data
        sys.stdout.write(promt.format(f"set 'trset' to {train_data}"))
    if 'devset' not in training_settings:
        dev_data = fmt_data.format('dev')
        checkExist('f', dev_data)
        training_settings['devset'] = dev_data
        sys.stdout.write(promt.format(f"set 'devset' to {dev_data}"))
    if 'world-size' not in training_settings:
        training_settings['world-size'] = 1
    if 'rank' not in training_settings:
        training_settings['rank'] = 0
    if 'dir' not in training_settings:
        training_settings['dir'] = args.expdir
        sys.stdout.write(promt.format(f"set 'dir' to {args.expdir}"))
    if 'workers' not in training_settings:
        training_settings['workers'] = 2
        sys.stdout.write(promt.format(f"set 'workers' to 2"))
    if 'dist-url' not in training_settings:
        training_settings['dist-url'] = f"tcp://localhost:{get_free_port()}"
        sys.stdout.write(promt.format(
            f"set 'dist-url' to {training_settings['dist-url']}"))

    mp_spawn(MainFunc, updateNamespaceFromDict(training_settings, Parser))


def priorResolvePath(dataset: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
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


def combineText(datasets: Union[str, List[str]], f_out: Optional[str] = None, seperator: str = ' ') -> str:
    """Combine text files of dataset(s) and return the combined file."""
    text_noid, text_withid = priorResolvePath(datasets)
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
                    line = line.split()
                    fo.write(seperator.join(line[1:]) + '\n')
    return f_out


def TrainTokenizer(f_hyper: str):
    checkExist('f', f_hyper)
    hyper_settings = readfromjson(f_hyper)

    assert 'data' in hyper_settings, f"missing property 'data': {f_hyper}"
    assert 'train' in hyper_settings[
        'data'], f"missing property 'train' in ['data']: {f_hyper}"
    assert 'tokenizer' in hyper_settings, f"missing property 'tokenizer': {f_hyper}"
    assert 'type' in hyper_settings[
        'tokenizer'], f"missing property 'type' in ['tokenizer']: {f_hyper}"
    if 'location' not in hyper_settings['tokenizer']:
        sys.stderr.write(
            f"missing property 'location' in ['tokenizer']: {f_hyper}\n")
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
    assert os.access(os.path.dirname(hyper_settings['tokenizer']['location']),
                     os.W_OK), f"['tokenizer']['location'] is not writable: '{hyper_settings['tokenizer']['location']}'"

    if 'lang' in hyper_settings['data']:
        # check if it's chinese-like languages
        iszh = ('zh' == hyper_settings['data']['lang'].split('-')[0])
    else:
        iszh = False
    seperator = '' if iszh else ' '

    f_corpus_tmp = combineText(
        hyper_settings['data']['train'], seperator=seperator)
    tokenizer_type = hyper_settings['tokenizer']['type']
    if 'property' not in hyper_settings['tokenizer']:
        hyper_settings['tokenizer']['property'] = {}

    try:
        import cat
    except ModuleNotFoundError:
        sys.path.append(os.getcwd())
    from cat.shared import tokenizer as tknz
    if tokenizer_type == 'SentencePieceTokenizer':
        sp_settings, (f_tokenizer, _) = resolve_sp_path(
            hyper_settings['tokenizer']['property'], os.path.basename(os.getcwd()), allow_making=True)
        sentencepiece_train(f_corpus_tmp, **sp_settings)
        hyper_settings['tokenizer']['property'] = sp_settings
        tokenizer = tknz.SentencePieceTokenizer(spmodel=f_tokenizer)
        hyper_settings['tokenizer']['property']['vocab_size'] = tokenizer.vocab_size
    elif tokenizer_type == 'JiebaTokenizer':
        # jieba tokenizer doesn't need training
        tokenizer = tknz.JiebaTokenizer(
            **hyper_settings['tokenizer']['property'])
    else:
        raise ValueError(f"Unknown type of tokenizer: {tokenizer_type}")

    tknz.save(tokenizer, hyper_settings['tokenizer']['location'])

    dumpjson(hyper_settings, f_hyper)
    os.remove(f_corpus_tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("expdir", type=str, help="Experiment directory.")
    parser.add_argument("--start_stage", dest='stage_beg',
                        type=int, default=1, help="Start stage of processing. Default: 1")
    parser.add_argument("--stop_stage", dest='stage_end',
                        type=int, default=-1, help="Stop stage of processing. Default: last stage.")
    parser.add_argument("--ngpu", type=int, default=-1,
                        help="Number of GPUs to be used.")
    parser.add_argument("--silient", action="store_true",
                        help="Disable detailed messages output.")

    args = parser.parse_args()
    s_beg = args.stage_beg
    s_end = args.stage_end
    if s_end == -1:
        s_end = float('inf')

    assert s_end >= 1, f"Invalid stop stage: {s_end}"
    assert s_beg >= 1 and s_beg <= s_end, f"Invalid start stage: {s_beg}"

    cwd = os.getcwd()
    checkExist('d', args.expdir)
    f_hyper_settings = os.path.join(args.expdir, 'hyper-p.json')
    checkExist('f', f_hyper_settings)
    initial_datainfo()
    datainfo = readfromjson(F_DATAINFO)
    if args.ngpu > -1:
        set_visible_gpus(args.ngpu)

    ############ Stage 1  Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 1 Tokenizer training"))
        fmt = "# Tokenizer trainin # {}\n" if not args.silient else ""
        hyper_settings = readfromjson(f_hyper_settings)
        if 'tokenizer' not in hyper_settings:
            sys.stderr.write(
                "warning: missing 'tokenizer' in hyper-setting, skip tokenizer training.")
        else:
            TrainTokenizer(f_hyper_settings)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 2 Pickle data"))
        fmt = "# Pickle data # {}\n" if not args.silient else ""

        hyper_settings = readfromjson(f_hyper_settings)
        assert 'data' in hyper_settings, f"missing property 'data': {f_hyper_settings}"
        if 'lang' in hyper_settings['data']:
            # check if it's chinese-like languages
            iszh = ('zh' == hyper_settings['data']['lang'].split('-')[0])
        else:
            iszh = False

        try:
            import cat
        except ModuleNotFoundError:
            sys.path.append(os.getcwd())
        from cat.shared import tokenizer as tknz
        if 'tokenizer' not in hyper_settings:
            sys.stderr.write(
                f"warning: missing property 'tokenizer', assume the ground truth text files as tokenized ones.")
            istokenized = True
            tokenizer = None
        else:
            # load tokenizer from file
            assert 'location' in hyper_settings[
                'tokenizer'], f"missing property 'location' in ['tokenizer']: {f_hyper_settings}"

            f_tokenizer = hyper_settings['tokenizer']['location']
            checkExist('f', f_tokenizer)
            tokenizer = tknz.load(f_tokenizer)
            istokenized = False

        data_settings = hyper_settings['data']
        if 'filter' not in data_settings:
            data_settings['filter'] = None

        d_pkl = os.path.join(args.expdir, 'pkl')
        os.makedirs(d_pkl, exist_ok=True)
        for dataset in ['train', 'dev', 'test']:
            if dataset not in data_settings:
                sys.stderr.write(
                    f"warning: missing '{dataset}' in ['data'], skip.\n")
                continue
            sys.stdout.write(fmt.format(f"parsing {dataset} data..."))

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

            parsingData(
                [_data['scp'] for _data in f_data],
                [_data['trans'] for _data in f_data],
                f_out=os.path.join(d_pkl, dataset+'.pkl'),
                filter=filter, tokenizer=tokenizer, iszh=iszh)
            del f_data

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 3 NN training"))
        fmt = "# NN training # {}\n" if not args.silient else ""
        try:
            import cat
        except ModuleNotFoundError:
            sys.path.append(cwd)

        hyper_settings = readfromjson(f_hyper_settings)
        if 'topo' not in hyper_settings:
            hyper_settings['topo'] = 'rnnt'
            sys.stdout.write(fmt.format(f"set 'topo' to 'rnnt'"))

        if hyper_settings['topo'] == 'rnnt':
            from cat.rnnt.train import RNNTParser
            from cat.rnnt.train import main as RNNTMain

            TrainNNModel(args, hyper_settings, f_hyper_settings, os.path.join(
                args.expdir, 'pkl/{}.pkl'), RNNTParser(), RNNTMain, fmt)
        elif hyper_settings['topo'] == 'ctc':
            from cat.ctc.train import CTCParser
            from cat.ctc.train import main as CTCMain
            TrainNNModel(args, hyper_settings, f_hyper_settings, os.path.join(
                args.expdir, 'pkl/{}.pkl'), CTCParser(), CTCMain, fmt)
        else:
            raise ValueError(fmt.format(
                f"Unknown topology: {hyper_settings['topo']}, expect one of ['rnnt', 'ctc']"))

    ############ Stage 4  Decode ############
    if s_beg <= 4 and s_end >= 4:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 4 Decode"))
        fmt = "# Decode # {}\n" if not args.silient else ""
        import re
        import torch
        try:
            import cat
        except ModuleNotFoundError:
            sys.path.append(cwd)

        hyper_settings = readfromjson(f_hyper_settings)
        assert 'inference' in hyper_settings, f"missing property 'inference': {f_hyper_settings}"
        assert 'data' in hyper_settings, f"missing property 'data': {f_hyper_settings}"
        assert 'test' in hyper_settings[
            'data'], "missing property 'test' in ['data']: {f_hyper_settings}"
        if 'topo' not in hyper_settings:
            hyper_settings['topo'] = 'rnnt'
            sys.stdout.write(fmt.format(f"set 'topo' to 'rnnt'"))

        if hyper_settings['topo'] not in ['rnnt', 'ctc']:
            raise ValueError(
                f"Unknown topology: {hyper_settings['topo']}, expect one of ['rnnt', 'ctc']")

        inference_settings = hyper_settings['inference']
        checkdir = os.path.join(args.expdir, 'checks')
        checkExist('d', checkdir)

        # decode
        assert 'decode' in inference_settings, "missing 'decode' in hyper-p['inference']"
        decode_settings = inference_settings['decode']

        if 'resume' in decode_settings:
            sys.stdout.write(fmt.format(
                "setting 'resume' in decoding would overwrite the model averaging settings."))
            try:
                checkExist('f', decode_settings['resume'])
                checkpoint = decode_settings['resume']
            except FileNotFoundError:
                checkExist('f', os.path.join(
                    checkdir, decode_settings['resume']))
                checkpoint = os.path.join(checkdir, decode_settings['resume'])
            # rm dirname and '.pt'
            suffix_avgmodel = os.path.basename(checkpoint)[:-3]
        else:
            # model averaging
            if 'avgmodel' not in inference_settings:
                suffix_avgmodel = 'best-1'
                checkpoint = os.path.join(checkdir, 'bestckpt.pt')
            else:
                assert 'avgmodel' in inference_settings, "missing 'avgmodel' in hyper-p['inference']"

                avgmodel_settings = inference_settings['avgmodel']
                assert 'mode' in avgmodel_settings, "missing 'mode' in hyper-p['inference']['avgmodel']"
                assert 'num' in avgmodel_settings, "missing 'num' in hyper-p['inference']['avgmodel']"
                avg_mode, avg_num = avgmodel_settings['mode'], avgmodel_settings['num']

                from avgmodel import find_n_best, average_checkpoints
                suffix_avgmodel = f"{avg_mode}-{avg_num}"
                if avg_mode == 'best':
                    f_check_list = find_n_best(checkdir, avg_num)
                elif avg_mode == 'last':
                    pattern = re.compile(r"checkpoint[.]\d{3}[.]pt")
                    f_check_all = [os.path.join(checkdir, _f) for _f in os.listdir(
                        checkdir) if pattern.search(_f) is not None]
                    if len(f_check_all) < avg_num:
                        raise RuntimeError(
                            f"trying to do model averaging {avg_num} over {len(f_check_all)} checkpoint.")
                    f_check_list = sorted(f_check_all, reverse=True)[:avg_num]
                    del f_check_all
                else:
                    raise NotImplementedError(
                        f"Unknown model averaging mode {avg_mode}")
                checkpoint = os.path.join(checkdir, suffix_avgmodel+'.pt')
                params = average_checkpoints(f_check_list)
                torch.save(params, checkpoint)

            _hyper = readfromjson(f_hyper_settings)
            _hyper['inference']['decode']['resume'] = checkpoint
            dumpjson(_hyper, f_hyper_settings)

        checkExist('f', checkpoint)
        decode_settings['resume'] = checkpoint
        sys.stdout.write(fmt.format(
            f"set 'resume' to {checkpoint}"))

        if 'config' not in decode_settings:
            decode_settings['config'] = os.path.join(
                args.expdir, 'config.json')
            sys.stdout.write(fmt.format(
                f"set 'config' to {decode_settings['config']}"))
            checkExist('f', decode_settings['config'])
        sys.stdout.write(fmt.format(
            "\n"
            "Ensure that the tokenizer you specify is correct.\n"
            "If you're doing rescoring, the tokenizer should be set as the LM one."
        ))
        if 'tokenizer' not in decode_settings:
            assert 'tokenizer' in hyper_settings, (
                "\nyou should set at least one of:\n"
                f"1. set 'tokenizer' and ['tokenizer']['location'] in {f_hyper_settings}\n"
                f"2. set 'tokenizer' in ['inference']['decode'] in {f_hyper_settings}\n")
            decode_settings['tokenizer'] = hyper_settings['tokenizer']['location']
        if 'nj' not in decode_settings:
            decode_settings['nj'] = os.cpu_count()
            sys.stdout.write(fmt.format(
                f"set 'nj' to {decode_settings['nj']}"))
        if hyper_settings['topo'] == 'rnnt' and (
                'alpha' in decode_settings and decode_settings['alpha'] is not None):

            if 'lmdir' not in inference_settings and \
                    ('lm-config' not in decode_settings or 'lm-check' not in decode_settings):
                sys.stderr.write(
                    "\n"
                    "To use external LM with RNN-T topo, at least one option is required:\n"
                    "  1 (higer priority): set 'lmdir' in hyper-p['inference'];\n"
                    "  2: set both 'lm-config' and 'lm-check' in hyper-p['inference']['decode']\n\n")
                exit(1)

            if 'beta' in decode_settings:
                beta_suffix = f"-{decode_settings['beta']}"
            else:
                beta_suffix = ''
            if 'lmdir' in inference_settings:
                lmdir = inference_settings['lmdir']
                checkExist('d', lmdir)
                decode_settings['lm-config'] = os.path.join(
                    lmdir, 'config.json')
                checkExist('f', decode_settings['lm-config'])
                decode_settings['lm-check'] = os.path.join(
                    lmdir, 'checks/bestckpt.pt')
                sys.stdout.write(fmt.format(
                    f"set 'lm-config' to {decode_settings['lm-config']}"))
                sys.stdout.write(fmt.format(
                    f"set 'lm-check' to {decode_settings['lm-check']}"))

            if 'rescore' in decode_settings and decode_settings['rescore']:
                suffix_lm = f"lm-rescore-{decode_settings['alpha']}{beta_suffix}"
            else:
                suffix_lm = f"lm-fusion-{decode_settings['alpha']}{beta_suffix}"

        elif hyper_settings['topo'] == 'ctc' and 'lm-path' in decode_settings:
            checkExist('f', decode_settings['lm-path'])
            if 'alpha' in decode_settings:
                alpha = str(decode_settings['alpha'])
            else:
                alpha = 'default'
            if 'beta' in decode_settings:
                beta = str(decode_settings['beta'])
            else:
                beta = 'default'
            suffix_lm = f"lm-fusion-{alpha}-{beta}"
        else:
            suffix_lm = "nolm"

        if 'output_prefix' not in decode_settings:
            decodedir = os.path.join(args.expdir, 'decode')
            os.makedirs(decodedir, exist_ok=True)
            if hyper_settings['topo'] == 'rnnt':
                f_text = f"rnnt-{decode_settings['beam-size']}_algo-{decode_settings['algo']}_{suffix_lm}_{suffix_avgmodel}"
            else:
                f_text = f"ctc-{decode_settings['beam-size']}_{suffix_lm}_{suffix_avgmodel}"
            decode_out_prefix = os.path.join(decodedir, f_text)
            sys.stdout.write(fmt.format(
                f"set 'output_prefix' to {decode_out_prefix}"))
        else:
            decode_out_prefix = decode_settings['output_prefix']
        if hyper_settings['topo'] == 'rnnt' and 'algo' in decode_settings \
                and decode_settings['algo'] == 'alsd' and 'umax-portion' not in decode_settings:
            # trying to compute a reasonable portion value from trainset
            from cat.shared.data import KaldiSpeechDataset
            import numpy as np
            f_pkl = os.path.join(args.expdir, f'pkl/train.pkl')
            if os.path.isfile(f_pkl):
                # since we using conformer, there's a 1/4 subsapling, if it's not, modify that
                if "subsample" in inference_settings:
                    sub_factor = inference_settings['subsample']
                    assert sub_factor > 1, f"can't deal with 'subsample'={sub_factor} in hyper-p['inference']"
                    sys.stdout.write(fmt.format(
                        f"resolving portion data from train set, might takes a while."))
                    dataset = KaldiSpeechDataset(f_pkl)
                    lt = np.asarray(dataset.get_seq_len()).astype(np.float64)
                    # get label lengths
                    ly = np.zeros_like(lt)
                    for i in range(len(dataset)):
                        _, label = dataset[i]
                        ly[i] = label.size(0)

                    lt /= sub_factor
                    normal_ly = ly/lt
                    portion = np.mean(normal_ly) + 5 * np.std(normal_ly)
                    del lt
                    del ly
                    del normal_ly
                    decode_settings['umax-portion'] = portion
                    orin_hyper_setting = readfromjson(f_hyper_settings)
                    orin_hyper_setting['inference']['decode']['umax-portion'] = portion
                    dumpjson(orin_hyper_setting, f_hyper_settings)
                    sys.stdout.write(fmt.format(
                        f"set 'umax-portion' to {portion}"))
            else:
                sys.stderr.write(
                    f"warning: no training set found in {args.expdir}. Skip counting 'umax-portion'")
        testsets = hyper_settings['data']['test']
        if isinstance(testsets, str):
            testsets = [testsets]
        f_scps = [datainfo[_set]['scp'] for _set in testsets]
        checkExist('f', f_scps)

        if hyper_settings['topo'] == 'rnnt':
            from cat.rnnt.decode import DecoderParser
            from cat.rnnt.decode import main as DecoderMain
        elif hyper_settings['topo'] == 'ctc':
            from cat.ctc.decode import DecoderParser
            from cat.ctc.decode import main as DecoderMain
        else:
            raise RuntimeError(
                f"Unknown topology: {hyper_settings['topo']}")

        for _set, scp in zip(testsets, f_scps):
            decode_settings['output_prefix'] = decode_out_prefix+f'_{_set}'
            decode_settings['input_scp'] = scp
            sys.stdout.write(fmt.format(
                f"{scp} -> {decode_settings['output_prefix']}"))

            # FIXME: this canonot be spawned via mp_spawn, otherwise error would be raised
            #        possibly due to the usage of mp.Queue
            if os.path.isfile(decode_settings['output_prefix']):
                sys.stdout.write(fmt.format(
                    f"{decode_settings['output_prefix']} exists, skip this one."))
                continue

            DecoderMain(updateNamespaceFromDict(
                decode_settings, DecoderParser()))

        # compute wer/cer
        from wer import WERParser
        from wer import main as WERMain
        assert 'er' in inference_settings, "missing 'er' in hyper-p['inference']"
        err_settings = inference_settings['er']
        assert 'mode' in err_settings, "missing 'mode' in hyper-p['inference']['er']"

        f_texts = [datainfo[_set]['trans'] for _set in testsets]
        checkExist('f', f_texts)

        if 'oracle' not in err_settings:
            err_settings['oracle'] = True
            sys.stdout.write(fmt.format(f"set 'oracle' to True"))

        if 'stripid' not in err_settings:
            err_settings['stripid'] = True
            sys.stdout.write(fmt.format(f"set 'stripid' to True"))

        err_settings.update({
            'cer': True if err_settings['mode'] == 'cer' else False
        })
        del err_settings['mode']

        for _set, _text in zip(testsets, f_texts):
            err_settings.update({
                'gt': _text,
                'hy': decode_out_prefix+f'_{_set}'
            })

            if err_settings['oracle']:
                err_settings['oracle'] = False
                # compute non-oracle WER/CER
                print(_set, end='\t')
                mp_spawn(WERMain, updateNamespaceFromDict(err_settings, WERParser(), [
                    err_settings['gt'], err_settings['hy']]))
                err_settings['oracle'] = True
                err_settings['hy'] += '.nbest'

            print(_set, end='\t')
            mp_spawn(WERMain, updateNamespaceFromDict(err_settings, WERParser(), [
                err_settings['gt'], err_settings['hy']]))
