"""Ported from run_rnnt.sh, rewrote with python

Uage:
    python utils/asr_process.py
"""


import os
import sys
import json
import uuid
import pickle
import argparse
from typing import Union, Literal, List, Tuple, Optional, Callable, Dict
from multiprocessing import Process


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
    f_data = f_out+'.npy'
    f_linfo = f_out + '.linfo'

    if os.path.isfile(f_out):
        sys.stderr.write("warning: parsingData() "
                         f"file exists: {f_out}, "
                         "rm it if you want to update the data.\n\n")
        checkExist('f', f_data)
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
    cnt_tokens = 0
    linfo = np.empty(total_utts, dtype=np.int64)
    _seeks = np.empty(total_utts, dtype=np.int64)
    _keys = []
    idx = 0
    cnt_rm = 0

    with open(f_data, 'wb') as fo_bin:
        for n, _f_scp in enumerate(f_scps):
            with open(_f_scp, 'r') as fi_scp:
                for line in tqdm(fi_scp, total=num_utts[n]):
                    key, loc_ark = line.split()
                    tag = labels[key]
                    feature = kaldiio.load_mat(
                        loc_ark, fd_dict=f_opened)   # type:np.ndarray

                    if feature.shape[0] < l_min or feature.shape[0] > l_max:
                        cnt_rm += 1
                        continue

                    linfo[idx] = feature.shape[0]
                    _seeks[idx] = fo_bin.tell()
                    _keys.append(key)
                    np.save(fo_bin, np.asarray(feature, dtype=np.float32))
                    np.save(fo_bin, np.asarray(tag, dtype=np.int64))

                    cnt_frames += feature.shape[0]
                    cnt_tokens += tag.shape[0]
                    idx += 1

    linfo = linfo[:idx]
    _seeks = _seeks[:idx]

    for f in f_opened.values():
        f.close()

    with open(f_out, 'wb') as fo:
        pickle.dump(os.path.basename(f_data), fo)
        pickle.dump(_seeks, fo)
        pickle.dump(_keys, fo)

    # save length info in case of further usage
    with open(f_linfo, 'wb') as fo:
        pickle.dump(linfo, fo)

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


def expandPath(f_type: Literal['t', 's'], s_list: Union[str, List[str]], cwd: Optional[str] = None) -> List[str]:
    """Expand the dataset to path of CAT data location.

    Args:
        f_type (str)          : 't': text, 's': scp
        s_list (list, str)    : list of dataset(s), can be string
        cwd    (str, optional): location of current working directory
    """
    if cwd is None:
        cwd = os.getcwd()

    cat_data = f'../../tools/CAT/egs/{os.path.basename(cwd)}/data'
    checkExist('d', cat_data)
    if f_type == 't':
        fmt = os.path.join(cat_data, '{}/text')
    elif f_type == 's':
        fmt = os.path.join(cat_data, 'all_ark/{}.scp')
    else:
        raise RuntimeError(f"expandPath: Unknown expand type '{f_type}'")

    if isinstance(s_list, str):
        return [fmt.format(s_list)]
    else:
        return [fmt.format(s) for s in s_list]


def recursive_rpl(src_dict: dict, target_key: str, rpl_val):
    if not isinstance(src_dict, dict):
        return

    if target_key in src_dict:
        src_dict[target_key] = rpl_val
    else:
        for k, v in src_dict.items():
            recursive_rpl(v, target_key, rpl_val)


def updateNamespaceFromDict(src_dict: dict, parser: argparse.ArgumentParser, positionals: List[str] = []):
    args = argparse.Namespace()
    processed_dict = {}
    for k, v in src_dict.items():
        processed_dict[k.replace('-', '_')] = v
    args.__dict__.update(processed_dict)
    to_args = parser.parse_args(args=positionals, namespace=args)
    return to_args


def generate_visible_gpus(N: int) -> str:
    assert N >= 0
    return ','.join([str(i) for i in range(N)])


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

    if args.ngpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = \
            generate_visible_gpus(args.ngpu)

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


def priorResolvePath(dataset: Union[str, List[str]], local_dir: str = 'data/text') -> Tuple[List[str], List[str]]:
    """Resolve text file location for dataset.

    Args:
        dataset (str, list): dataset(s)

    Returns:
        (local_texts, outside_texts)
    """
    if isinstance(dataset, str):
        dataset = [dataset]

    local_text = []
    outside_text = []
    for _set in dataset:
        if os.path.isfile(_set):
            local_text.append(_set)
        else:
            _text = os.path.join(local_dir, _set)
            if os.path.isfile(_text):
                local_text.append(_text)
            else:
                outside_text.append(_set)

    if outside_text != []:
        try:
            outside_text = expandPath('t', outside_text)
            checkExist('f', outside_text)
        except FileNotFoundError as fe:
            print(fe)
            print("Resolve outside path as empty.")
            outside_text = []
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


def hyperParamSearch(
        decode_settings: dict,
        er_settings: dict,
        lm_config: str,
        lm_check: Union[str, None],
        dev_set: Optional[str] = None,
        topo: Literal['ctc', 'rnnt'] = 'rnnt',
        fmt: str = '{}',
        f_nbestlist: Optional[str] = None):
    """Search for hyper params alpha/beta for LM integration

    decode_settings : decode settings
    er_settings : wer/cer evaluation settings
    lm_config : language model configuration
    lm_check : checkpoint of model, can be none for ngram model.
    dev_set : dev set to evaluate the perf.
    topo : 'ctc' or 'rnnt'
    fmt : format string for hints
    f_nbestlist : existing n-best list file, optional.
    """

    if 'alpha' in decode_settings and 'beta' in decode_settings:
        return decode_settings['alpha'], decode_settings['beta']
    if dev_set is None:
        assert 'input_scp' in dev_set and 'gt' in er_settings, \
            "hyperParamSearch: dev set is required for hyper-param searching."
    else:
        decode_settings['input_scp'] = expandPath('s', dev_set)[0]
        er_settings['gt'] = expandPath('t', dev_set)[0]

    try:
        import cat
    except ModuleNotFoundError:
        sys.path.append(cwd)
    from wer import WERParser
    from wer import main as WERMain

    from cat.lm.rescore import (
        RescoreParser,
        main as RescoreMain
    )

    if topo == 'rnnt':
        from cat.rnnt.decode import DecoderParser
        from cat.rnnt.decode import main as DecoderMain
    elif topo == 'ctc':
        from cat.ctc.decode import DecoderParser
        from cat.ctc.decode import main as DecoderMain
    else:
        raise RuntimeError(f"Unknown topology: {topo}")

    cache_dir = '/tmp'
    if f_nbestlist is None:
        decode_settings['output_prefix'] = os.path.join(
            cache_dir, str(uuid.uuid4()))
        nbestlist = decode_settings['output_prefix']+'.nbest'
        decode_settings['alpha'] = None
        decode_settings['beta'] = None
        # generate nbest-list file
        sys.stdout.write(fmt.format(
            f"generate n-best list w/o LM at {nbestlist}"))
        DecoderMain(updateNamespaceFromDict(
            decode_settings, DecoderParser()))
        os.remove(decode_settings['output_prefix'])
    else:
        nbestlist = f_nbestlist

    rescore_setting = {
        'nbestlist': nbestlist,
        'config': lm_config,
        'resume': lm_check,
        'tokenizer': decode_settings['tokenizer'],
        'nj': decode_settings['nj'] if 'nj' in decode_settings else 16,
        'cpu': decode_settings['cpu'] if 'cpu' in decode_settings else True
    }

    def _evaluate():
        mapkey = f"{tuning_val[0]}:{tuning_val[1]}"
        if mapkey in searchout:
            return searchout[mapkey]

        rescore_setting.update({
            'alpha': tuning_val[0],
            'beta': tuning_val[1]
        })
        rescore_setting['output'] = os.path.join(
            cache_dir, str(uuid.uuid4()))
        print(f"Rescored output: {rescore_setting['output']}")
        print(
            f"Setting: alpha = {tuning_val[0]:.2f} | beta = {tuning_val[1]:.2f}")

        RescoreMain(updateNamespaceFromDict(
            rescore_setting, RescoreParser(), [rescore_setting['nbestlist'], rescore_setting['output']]))

        er_settings['hy'] = rescore_setting['output']
        # ignore 'oracle' setting
        if 'oracle' in er_settings:
            del er_settings['oracle']

        wer = WERMain(updateNamespaceFromDict(er_settings, WERParser(), [
                      er_settings['gt'], er_settings['hy']]))

        os.remove(rescore_setting['output'])
        searchout[mapkey] = wer
        return wer

    def _search(
            s_range: Tuple[int, int],
            _interval: float,
            _val: str,
            metric: Literal[0, 1],  # 0 for alpha, 1 for beta
            result: List[Union[float, None]] = [None, None]
    ) -> Tuple[float, Union[None, float]]:
        result = list(result)
        lower, upper = s_range
        best_m = (0., None)
        while upper - lower >= _interval:
            if result[0] is None:
                tuning_val[metric] = lower
                result[0] = _evaluate()[_val]

            if result[1] is None:
                tuning_val[metric] = upper
                result[1] = _evaluate()[_val]

            if result[0] > result[1]:
                result[0] = None
                lower = (lower+upper)/2
                best_m = (upper, result[1])
            else:
                result[1] = None
                upper = (lower+upper)/2
                best_m = (lower, result[0])

        tuning_val[metric] = best_m[0]
        return best_m[1]

    val = 'wer'
    alpha_range = [0.0, 1.0]
    alpha_interval = 0.02
    beta_range = [-1.0, 1.0]
    beta_interval = 0.2
    tuning_val = [0, sum(beta_range)/2]
    searchout = {}  # type: Dict[str, Dict[str, Union[float, int]]]

    def update_tunning_val():
        # e.g. tunning_val = 0.5:-0.3
        tuning_val = min(searchout.keys(),
                         key=lambda k: searchout[k][val])  # type: str
        return [float(x) for x in tuning_val.split(':')]

    n_iter = 1
    last_val = [None, None]
    while last_val != tuning_val:
        print(
            f"\nSearching iter: {n_iter} | alpha {alpha_range} | beta {beta_range}")
        last_val = tuning_val.copy()
        # stage 1: fix beta, search alpha in range
        _search(alpha_range, alpha_interval, val, 0)
        tuning_val = update_tunning_val()
        # stage 2: fix alpha, search beta in range
        _search(beta_range, beta_interval, val, 1)
        tuning_val = update_tunning_val()

        cur_alpha, cur_beta = tuning_val

        if cur_alpha in alpha_range:
            da = (alpha_range[1] - alpha_range[0])/2
            alpha_range = [cur_alpha-da, cur_alpha+da]

        if cur_beta in beta_range:
            db = (beta_range[1] - beta_range[0]) / 2
            beta_range = [cur_beta-db, cur_beta+db]
        n_iter += 1

    if f_nbestlist is None:
        os.remove(nbestlist)

    del _evaluate
    del _search
    del update_tunning_val

    return tuple(tuning_val), searchout[f"{tuning_val[0]}:{tuning_val[1]}"]


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
                    f"warning: missing '{dataset}' in ['data'], skip.")
                continue
            sys.stdout.write(fmt.format(f"parsing {dataset} data..."))
            f_texts = expandPath('t', data_settings[dataset], cwd)
            f_scps = expandPath('s', data_settings[dataset], cwd)
            if dataset == 'train':
                filter = data_settings['filter']
            else:
                filter = None

            parsingData(f_scps, f_texts,
                        f_out=os.path.join(d_pkl, dataset+'.pkl'),
                        filter=filter, tokenizer=tokenizer, iszh=iszh)

        if not istokenized:
            # generate word prefix tree from train set
            from word_prefix_tree import WordPrefixParser
            from word_prefix_tree import main as WPTMain
            f_text = expandPath('t', data_settings['train'], cwd)[0]
            wpt_settings = {
                'stripid': True,
                'output': os.path.join(d_pkl, 'wpt.pkl')}

            mp_spawn(WPTMain, updateNamespaceFromDict(wpt_settings, WordPrefixParser(), [
                f_text, hyper_settings['tokenizer']['location']]))

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

        if_search_hyper = False
        if 'search-hyper' in inference_settings and inference_settings['search-hyper']:
            sys.stdout.write(fmt.format("enable hyper-param searching."))
            if_search_hyper = True

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

        if if_search_hyper:
            if 'alpha' not in decode_settings:
                decode_settings['alpha'] = None
            if 'beta' not in decode_settings:
                decode_settings['beta'] = None

            sys.stdout.write(fmt.format(
                "in case you didn't backup the setting:\n"
                f"current setting: alpha = {decode_settings['alpha']} | beta = {decode_settings['beta']}"))

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
            if_search_hyper or (
                'alpha' in decode_settings and decode_settings['alpha'] is not None)):

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
            if if_search_hyper:
                lm_info = {
                    'config': decode_settings['lm-config'],
                    'check': decode_settings['lm-check']}
        elif hyper_settings['topo'] == 'ctc' and (
                if_search_hyper or 'lm-path' in decode_settings):
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
            if if_search_hyper:
                dir_lm = os.path.dirname(decode_settings['lm-path'])
                lm_conf = os.path.join(dir_lm, 'config.json')
                checkExist('f', lm_conf)
                lm_info = {
                    'config': lm_conf,
                    'check': None}
                del dir_lm
                del lm_conf
        elif if_search_hyper:
            raise RuntimeError(
                "'search-hyper' = true but not configure LM.")
        else:
            suffix_lm = "nolm"

        if 'output_prefix' not in decode_settings:
            decodedir = os.path.join(args.expdir, 'decode')
            os.makedirs(decodedir, exist_ok=True)
            if hyper_settings['topo'] == 'rnnt':
                f_text = f"rnnt-{decode_settings['beam_size']}_algo-{decode_settings['algo']}_{suffix_lm}_{suffix_avgmodel}"
            else:
                f_text = f"ctc-{decode_settings['beam_size']}_{suffix_lm}_{suffix_avgmodel}"
            decode_out_prefix = os.path.join(decodedir, f_text)
            sys.stdout.write(fmt.format(
                f"set 'output_prefix' to {decode_out_prefix}"))
        else:
            decode_out_prefix = decode_settings['output_prefix']
        if hyper_settings['topo'] == 'rnnt' and 'algo' in decode_settings \
                and decode_settings['algo'] == 'alsd' and 'umax-portion' not in decode_settings:
            # trying to compute a reasonable portion value from trainset
            from cat.shared.data import ModifiedSpeechDataset
            import numpy as np
            f_pkl = os.path.join(args.expdir, f'pkl/train.pkl')
            if os.path.isfile(f_pkl):
                # since we using conformer, there's a 1/4 subsapling, if it's not, modify that
                if "subsample" in inference_settings:
                    sub_factor = inference_settings['subsample']
                    assert sub_factor > 1, f"can't deal with 'subsample'={sub_factor} in hyper-p['inference']"
                    sys.stdout.write(fmt.format(
                        f"resolving portion data from train set, might takes a while."))
                    dataset = ModifiedSpeechDataset(f_pkl)
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
        f_scps = expandPath('s', testsets, cwd)
        checkExist('f', f_scps)

        if not if_search_hyper:
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

        f_texts = expandPath('t', testsets, cwd)
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

        if if_search_hyper:
            if len(testsets) > 1:
                sys.stdout.write(fmt.format(
                    "found more than 1 evalute sets, take the first to do searching."))
                sys.stdout.write(fmt.format(f"{testsets} -> {testsets[0]}"))
            eval_set = testsets[0]
            del decode_settings['alpha']
            del decode_settings['beta']
            if 'nbestlist' in inference_settings:
                checkExist('f', inference_settings['nbestlist'])
                sys.stdout.write(fmt.format(
                    f"you specify 'nbestlist'={inference_settings['nbestlist']}, ensure the first test set match this file."))
            else:
                inference_settings['nbestlist'] = None

            (_alpha, _beta), werr = hyperParamSearch(
                decode_settings, err_settings,
                lm_info['config'], lm_info['check'],
                eval_set, hyper_settings['topo'], fmt, inference_settings['nbestlist'])
            sys.stdout.write(fmt.format(
                f"found best setting: \n"
                f"{werr['string']}\talpha = {_alpha} | beta = {_beta}"))

            sys.stdout.write(fmt.format(
                f"write back setting to {f_hyper_settings}"))

            _tmp_setting = readfromjson(f_hyper_settings)
            _tmp_setting['inference']['search-hyper'] = False
            _tmp_setting['inference']['decode']['alpha'] = _alpha
            _tmp_setting['inference']['decode']['beta'] = _beta
            dumpjson(_tmp_setting, f_hyper_settings)
            sys.stdout.write(fmt.format(
                "settings have been updated, re-run this script."))
            exit(0)

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
