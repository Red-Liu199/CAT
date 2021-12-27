"""Ported from run_rnnt.sh, rewrote with python

Uage:
    python utils/asr_process.py
"""


import os
import json
import uuid
import pickle
import argparse
from typing import Union, Literal, List, Tuple, Optional, Callable
from multiprocessing import Process


def mp_spawn(target: Callable, args: Union[tuple, argparse.Namespace]):
    """Spawn a new process to execute the target function with given args."""
    if isinstance(args, argparse.Namespace):
        args = (args, )
    worker = Process(target=target, args=args)

    worker.start()
    worker.join()
    if worker.exitcode is not None and worker.exitcode != 0:
        raise RuntimeError("worker unexpectedly terminated.")


def resolve_sp_path(config: dict, prefix: Optional[str] = None, allow_making: bool = False):
    if 'model_prefix' in config:
        spdir = os.path.dirname(config['model_prefix'])
        if not os.path.isdir(spdir):
            print(
                f"Warning: trying to resolve from an empty directory: {spdir}")
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
    # available options https://github.com/google/sentencepiece/blob/master/doc/options.md
    spm.SentencePieceTrainer.Train(**DEFAULT_SETTINGS)


def parsingData(f_scp: str, f_label: str, f_out: str, filter: Optional[str] = None, spm=None, iszh: bool = False):
    """Parsing audio feature and text label into pickle file.

    Args:
        f_scp   (str): Kaldi-like-style .scp file.
        f_label (str): Pure text file include utterance id and sentence labels. Split by space.
        f_out   (str): Ouput pickle file location.
        filter (str, optional): identifier for filtering out seqs with unqualified length. 
            such as '100:2000' means remove those whose length is shorter than 100 or longer than 2000. Default: None
        spm (sentencepiece.SentencePieceProcessor optional): If `spm` is None, lines in `f_label` MUST be token indices, 
            otherwise it should be text.
    """
    import kaldiio
    import numpy as np
    from tqdm import tqdm

    checkExist('f', [f_scp, f_label])
    checkExist('d', os.path.dirname(f_out))

    pkl_data = []
    cnt_rm = 0
    l_min = 1
    l_max = float('inf')
    if filter is not None:
        assert ':' in filter, f"parsingData: invalid filter format {filter}"
        l_bound, u_bound = (i for i in filter.split(':'))
        if l_bound != '':
            l_min = int(l_bound)
        if u_bound != '':
            l_max = int(u_bound)

    with open(f_label, 'r') as fi_label:
        labels = fi_label.readlines()
    labels = [l.split() for l in labels]
    if spm is None:
        labels = {l[0]: np.asarray([int(i) for i in l[1:]]) for l in labels}
    else:
        if iszh:
            labels = {l[0]: np.asarray(spm.encode(''.join(l[1:])))
                      for l in labels}
        else:
            labels = {l[0]: np.asarray(spm.encode(' '.join(l[1:])))
                      for l in labels}

    total_lines = sum(1 for _ in open(f_scp, 'r'))
    assert len(
        labels) == total_lines, f"parsingData: f_scp and f_label should match on the # lines, instead {total_lines} {len(labels)}"

    f_opened = {}
    cnt_frames = 0
    cnt_tokens = 0
    with open(f_scp, 'r') as fi_scp:
        for line in tqdm(fi_scp, total=total_lines):
            key, loc_ark = line.split()
            tag = labels[key]
            feature = kaldiio.load_mat(
                loc_ark, fd_dict=f_opened)   # type:np.ndarray

            if feature.shape[0] < l_min or feature.shape[0] > l_max:
                cnt_rm += 1
                continue

            pkl_data.append([key, loc_ark, tag])
            cnt_frames += feature.shape[0]
            cnt_tokens += tag.shape[0]

    for f in f_opened.values():
        f.close()

    with open(f_out, 'wb') as fo:
        pickle.dump(pkl_data, fo)

    print(f"parsingData: remove {cnt_rm} unqualified sequences.")
    print(
        f"...# frames: {cnt_frames} | # tokens: {cnt_tokens} | # seqs: {len(pkl_data)}")


def combinePickle(f_in: Union[List[str], str], f_out: str, rm_src: bool = False):
    """Combine several pickle files into one.

    Args:
        f_in   (list, str)     : pickle file(s) to be merged.
        f_out  (str)           : destination file.
        rm_src (bool, optional): specify whether rm original files.
    """
    import shutil

    if isinstance(f_in, str):
        f_in = [f_in]

    checkExist('f', f_in)
    checkExist('d', os.path.dirname(f_out))

    if len(f_in) == 1:
        if rm_src:
            shutil.move(f_in[0], f_out)
        else:
            shutil.copy(f_in[0], f_out)
        return

    datasets = []
    for f_pkl in f_in:
        print("combinePickle: merging {}".format(f_pkl))
        with open(f_pkl, 'rb') as fi:
            datasets += pickle.load(fi)

    print("combinePickle: export to {}".format(f_out))
    with open(f_out, 'wb') as fo:
        pickle.dump(datasets, fo)


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


def NNTrain(
        args: argparse.Namespace,
        settings: dict,
        f_hyper_p: str,
        fmt_data: str,
        Parser: argparse.ArgumentParser,
        MainFunc: Callable[[argparse.Namespace], None],
        promt: str = '{}'):
    import subprocess

    assert 'sp' in settings, promt.format(f"missing 'sp' in hyper-p")
    assert 'train' in settings, promt.format("missing 'train' in hyper-p")

    _, (_, spvocab) = resolve_sp_path(settings['sp'])
    checkExist('f', spvocab)
    f_nnconfig = os.path.join(args.expdir, 'config.json')
    checkExist('f', f_nnconfig)

    if args.ngpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = \
            generate_visible_gpus(args.ngpu)

    with open(f_nnconfig, 'r') as fi:
        nnconfig = json.load(fi)
    # setup the num_classes to vocab_size
    with open(spvocab, 'r') as fi:
        n_vocab = sum(1 for _ in fi)
    # recursively search for 'num_classes'
    recursive_rpl(nnconfig, 'num_classes', n_vocab)
    with open(f_nnconfig, 'w') as fo:
        json.dump(nnconfig, fo, indent=4)

    if subprocess.run('command -v git', shell=True, capture_output=True).returncode != 0:
        print(promt.format("git command not found. Suppress saving git commit."))
    else:
        process = subprocess.run(
            "git log -n 1 --pretty=format:\"%H\"", shell=True, check=True, stdout=subprocess.PIPE)
        with open(f_hyper_p, 'r') as fi:
            orin_settings = json.load(fi)
        orin_settings['commit'] = process.stdout.decode('utf-8')
        with open(f_hyper_p, 'w') as fo:
            json.dump(orin_settings, fo, indent=4)

    training_settings = settings['train']
    if 'trset' not in training_settings:
        train_data = fmt_data.format('train')
        checkExist('f', train_data)
        training_settings['trset'] = train_data
        print(promt.format(f"set 'trset' to {train_data}"))
    if 'devset' not in training_settings:
        dev_data = fmt_data.format('dev')
        checkExist('f', dev_data)
        training_settings['devset'] = dev_data
        print(promt.format(f"set 'devset' to {dev_data}"))
    if 'world-size' not in training_settings:
        training_settings['world-size'] = 1
    if 'rank' not in training_settings:
        training_settings['rank'] = 0
    if 'dir' not in training_settings:
        training_settings['dir'] = args.expdir
        print(promt.format(f"set 'dir' to {args.expdir}"))
    if 'workers' not in training_settings:
        training_settings['workers'] = 2
        print(promt.format(f"set 'workers' to 2"))
    if 'dist-url' not in training_settings:
        training_settings['dist-url'] = f"tcp://localhost:{get_free_port()}"
        print(promt.format(
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


def SentencePieceTrain(
        settings: dict,
        f_hyper: str, promt: str = '{}'):
    assert 'data' in settings, promt.format(
        f"missing 'data' in hyper-setting file {f_hyper}")
    assert 'sp' in settings, promt.format(
        f"missing 'sp' in hyper-setting file {f_hyper}")
    f_corpus_tmp = os.path.join('/tmp', str(uuid.uuid4()))
    assert 'train' in settings['data'], promt.format(
        "missing 'train' in hyper-p['data']")
    if 'lang' in settings['data']:
        # check if it's chinese-like languages
        iszh = ('zh' == settings['data']['lang'].split('-')[0])
    else:
        iszh = False

    if iszh:
        seperator = ''
    else:
        seperator = ' '
    f_corpus_tmp = combineText(
        settings['data']['train'], seperator=seperator)
    sp_settings, (_, vocab) = resolve_sp_path(
        settings['sp'], os.path.basename(os.getcwd()), allow_making=True)
    sentencepiece_train(f_corpus_tmp, **sp_settings)

    checkExist('f', vocab)
    with open(vocab, 'r') as fi:
        n_vocab = sum(1 for _ in fi)
    sp_settings['vocab_size'] = n_vocab
    settings['sp'] = sp_settings
    with open(f_hyper, 'w') as fo:
        json.dump(settings, fo, indent=4)

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
    with open(f_hyper_settings, 'r') as fi:
        hyper_settings = json.load(fi)

    ############ Stage 1  Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        print("{0} {1} {0}".format("="*20, "Stage 1 Tokenizer training"))
        fmt = "Stage 1  Tokenizer training: {}"

        SentencePieceTrain(hyper_settings, f_hyper_settings, fmt)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        print("{0} {1} {0}".format("="*20, "Stage 2 Pickle data"))
        import sentencepiece as spm
        fmt = "Stage 2  Pickle data: {}"

        assert 'data' in hyper_settings, fmt.format(
            f"missing 'data' in hyper-setting file {f_hyper_settings}")
        assert 'sp' in hyper_settings, fmt.format(
            f"missing 'sp' in hyper-setting file {f_hyper_settings}")
        if 'lang' in hyper_settings['data']:
            # check if it's chinese-like languages
            iszh = ('zh' == hyper_settings['data']['lang'].split('-')[0])
        else:
            iszh = False

        _, (f_spm, _) = resolve_sp_path(hyper_settings['sp'])
        checkExist('f', f_spm)
        spmodel = spm.SentencePieceProcessor(model_file=f_spm)

        data_settings = hyper_settings['data']
        if 'filter' not in data_settings:
            data_settings['filter'] = None

        d_pkl = os.path.join(args.expdir, 'pkl')
        os.makedirs(d_pkl, exist_ok=True)
        for dataset in ['train', 'dev', 'test']:
            assert dataset in data_settings, fmt.format(
                f"missing '{dataset}' in hyper-p['data']")
            print(fmt.format(f" parsing {dataset} data..."))
            f_texts = expandPath('t', data_settings[dataset], cwd)
            f_scps = expandPath('s', data_settings[dataset], cwd)
            if dataset == 'train':
                filter = data_settings['filter']
            else:
                filter = None
            for f_s, f_t in zip(f_scps, f_texts):
                f_pkl_tmp = os.path.join('/tmp', str(uuid.uuid4()))
                parsingData(f_scp=f_s, f_label=f_t, f_out=f_pkl_tmp,
                            filter=filter, spm=spmodel, iszh=iszh)
            combinePickle(f_pkl_tmp, os.path.join(
                d_pkl, dataset+'.pkl'), rm_src=True)

        # generate word prefix tree from train set
        from word_prefix_tree import WordPrefixParser
        from word_prefix_tree import main as WPTMain
        f_text = expandPath('t', data_settings['train'], cwd)[0]
        wpt_settings = {
            'intext': f_text,
            'spmodel': f_spm,
            'stripid': True,
            'output': os.path.join(d_pkl, 'wpt.pkl')}

        mp_spawn(WPTMain, updateNamespaceFromDict(wpt_settings, WordPrefixParser(), [
            wpt_settings['intext'], wpt_settings['spmodel']]))

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        print("{0} {1} {0}".format("="*20, "Stage 3 NN training"))
        fmt = "Stage 3  NN training: {}"
        try:
            import cat
        except ModuleNotFoundError:
            import sys
            sys.path.append(cwd)

        if 'topo' not in hyper_settings:
            hyper_settings['topo'] = 'rnnt'
            print(fmt.format(f"set 'topo' to 'rnnt'"))

        if hyper_settings['topo'] == 'rnnt':
            from cat.rnnt.train import RNNTParser
            from cat.rnnt.train import main as RNNTMain

            NNTrain(args, hyper_settings, f_hyper_settings, os.path.join(
                args.expdir, 'pkl/{}.pkl'), RNNTParser(), RNNTMain, fmt)
        elif hyper_settings['topo'] == 'ctc':
            from cat.ctc.train import CTCParser
            from cat.ctc.train import main as CTCMain
            NNTrain(args, hyper_settings, f_hyper_settings, os.path.join(
                args.expdir, 'pkl/{}.pkl'), CTCParser(), CTCMain, fmt)
        else:
            raise ValueError(fmt.format(
                f"Unknown topology: {hyper_settings['topo']}, expect one of ['rnnt', 'ctc']"))

    ############ Stage 4  Decoding ############
    if s_beg <= 4 and s_end >= 4:
        print("{0} {1} {0}".format("="*20, "Stage 4 Decoding"))
        fmt = "Stage 4  Decoding: {}"
        import re
        import torch

        assert 'inference' in hyper_settings, fmt.format(
            f"missing 'inference' in hyper-setting file {f_hyper_settings}")
        assert 'data' in hyper_settings, fmt.format(
            f"missing 'data' in hyper-setting file {f_hyper_settings}")
        assert 'test' in hyper_settings['data'], fmt.format(
            "missing 'test' in hyper-p['data']")
        if 'topo' not in hyper_settings:
            hyper_settings['topo'] = 'rnnt'
            print(fmt.format(f"set 'topo' to 'rnnt'"))

        if hyper_settings['topo'] not in ['rnnt', 'ctc']:
            raise ValueError(fmt.format(
                f"Unknown topology: {hyper_settings['topo']}, expect one of ['rnnt', 'ctc']"))

        inference_settings = hyper_settings['inference']
        checkdir = os.path.join(args.expdir, 'checks')
        checkExist('d', checkdir)

        # decode
        assert 'decode' in inference_settings, fmt.format(
            "missing 'decode' in hyper-p['inference']")
        decode_settings = inference_settings['decode']

        if 'resume' in decode_settings:
            print(fmt.format(
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
                assert 'avgmodel' in inference_settings, fmt.format(
                    "missing 'avgmodel' in hyper-p['inference']")

                avgmodel_settings = inference_settings['avgmodel']
                assert 'mode' in avgmodel_settings, fmt.format(
                    "missing 'mode' in hyper-p['inference']['avgmodel']")
                assert 'num' in avgmodel_settings, fmt.format(
                    "missing 'num' in hyper-p['inference']['avgmodel']")
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
                        raise RuntimeError(fmt.format(
                            f"trying to do model averaging {avg_num} over {len(f_check_all)} checkpoint."))
                    f_check_list = sorted(f_check_all, reverse=True)[:avg_num]
                    del f_check_all
                else:
                    raise NotImplementedError(fmt.format(
                        f"Unknown model averaging mode {avg_mode}"))
                checkpoint = os.path.join(checkdir, suffix_avgmodel+'.pt')
                try:
                    params = average_checkpoints(f_check_list)
                    torch.save(params, checkpoint)
                except Exception as e:
                    if os.path.isfile(checkpoint):
                        print(fmt.format(e))
                        print(fmt.format(
                            f"Not found raw checkpoint files, use existing {checkpoint} instead."))
                    else:
                        raise RuntimeError(e)

            with open(f_hyper_settings, 'r') as fi:
                _hyper = json.load(fi)
            _hyper['inference']['decode']['resume'] = checkpoint
            with open(f_hyper_settings, 'w') as fo:
                json.dump(_hyper, fo, indent=4)

        checkExist('f', checkpoint)
        decode_settings['resume'] = checkpoint
        print(fmt.format(
            f"set 'resume' to {checkpoint}"))

        if 'config' not in decode_settings:
            decode_settings['config'] = os.path.join(
                args.expdir, 'config.json')
            print(fmt.format(f"set 'config' to {decode_settings['config']}"))
            checkExist('f', decode_settings['config'])
        if 'spmodel' not in decode_settings:
            assert 'sp' in hyper_settings, fmt.format(
                f"you should set at least one of 'sp' in hyper-p or 'spmodel' in hyper-p['inference']['decode']")
            _, (spmodel, _) = resolve_sp_path(hyper_settings['sp'])
            decode_settings['spmodel'] = spmodel
            print(fmt.format(
                f"set 'spmodel' to {decode_settings['spmodel']}"))
        if 'nj' not in decode_settings:
            decode_settings['nj'] = os.cpu_count()
            print(fmt.format(
                f"set 'nj' to {decode_settings['nj']}"))
        if hyper_settings['topo'] == 'rnnt' and \
                'alpha' in decode_settings and decode_settings['alpha'] > 0.0:
            assert 'lmdir' in inference_settings, fmt.format(
                "'alpha' > 0 in hyper-p['inference']['decode'] should be used with 'lmdir' in hyper-p['inference']")

            if 'beta' in decode_settings:
                beta_suffix = f"-{decode_settings['beta']}"
            else:
                beta_suffix = ''
            lmdir = inference_settings['lmdir']
            checkExist('d', lmdir)
            if 'lm-config' not in decode_settings:
                decode_settings['lm-config'] = os.path.join(
                    lmdir, 'config.json')
                print(fmt.format(
                    f"set 'lm-config' to {decode_settings['lm-config']}"))
                checkExist('f', decode_settings['lm-config'])
            if 'lm-check' not in decode_settings:
                decode_settings['lm-check'] = os.path.join(
                    lmdir, 'checks/bestckpt.pt')
                print(fmt.format(
                    f"set 'lm-check' to {decode_settings['lm-check']}"))
                # skip checking for n-gram model
                # checkExist('f', decode_settings['lm-check'])
            if 'rescore' in decode_settings and decode_settings['rescore']:
                suffix_lm = f"lm-rescore-{decode_settings['alpha']}{beta_suffix}"
            else:
                suffix_lm = f"lm-fusion-{decode_settings['alpha']}{beta_suffix}"
        elif hyper_settings['topo'] == 'ctc' and \
                'lm-path' in decode_settings:
            checkExist('f', decode_settings['lm-path'])
            if 'lm-alpha' in decode_settings:
                alpha = str(decode_settings['lm-alpha'])
            else:
                alpha = 'default'
            if 'lm-beta' in decode_settings:
                beta = str(decode_settings['lm-beta'])
            else:
                beta = 'default'
            suffix_lm = f"lm-fusion-{alpha}-{beta}"
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
            print(fmt.format(
                f"set 'output_prefix' to {decode_out_prefix}"))
        else:
            decode_out_prefix = decode_settings['output_prefix']
        if 'dist-url' not in decode_settings:
            decode_settings['dist-url'] = f"tcp://localhost:{get_free_port()}"
            print(fmt.format(
                f"set 'dist-url' to {decode_settings['dist-url']}"))
        if hyper_settings['topo'] == 'rnnt':
            if 'algo' in decode_settings and decode_settings['algo'] == 'alsd' and 'umax-portion' not in decode_settings:
                # trying to compute a reasonable portion value from trainset
                from cat.shared.data import SpeechDatasetPickle
                import numpy as np
                f_pkl = os.path.join(args.expdir, f'pkl/train.pkl')
                if os.path.isfile(f_pkl):
                    # since we using conformer, there's a 1/4 subsapling, if it's not, change that
                    if "subsample" in inference_settings:
                        sub_factor = inference_settings['subsample']
                        assert sub_factor > 1, fmt.format(
                            f"Can't deal with 'subsample'={sub_factor} in hyper-p['inference']")
                        print(fmt.format(
                            f"resolving portion data from train set, might takes a while."))
                        dataset = SpeechDatasetPickle(f_pkl)
                        lt = np.asarray(dataset.get_seq_len()
                                        ).astype(np.float64)
                        ly = []
                        for _, _, label in dataset.dataset:
                            ly.append(len(label))
                        assert len(lt) == len(ly), fmt.format(
                            f"Unknown condition {len(lt)} != {len(ly)}")
                        ly = np.asarray(ly).astype(np.float64)
                        lt /= sub_factor
                        normal_ly = ly/lt
                        portion = np.mean(normal_ly) + 5 * np.std(normal_ly)
                        del lt
                        del ly
                        del normal_ly
                        decode_settings['umax-portion'] = portion
                        with open(f_hyper_settings, 'r') as fi:
                            orin_hyper_setting = json.load(fi)
                        orin_hyper_setting['inference']['decode']['umax-portion'] = portion
                        with open(f_hyper_settings, 'w') as fo:
                            json.dump(orin_hyper_setting, fo, indent=4)
                        print(fmt.format(
                            f"set 'umax-portion' to {portion}"))
            # if 'word-tree' not in decode_settings and decode_settings['algo'] == 'alsd':
            #     pth_wpt = os.path.join(args.expdir, 'pkl/wpt.pkl')
            #     if os.path.isfile(pth_wpt):
            #         decode_settings['word-tree'] = pth_wpt
            #         print(fmt.format(f"set 'word-tree' to {pth_wpt}"))
            #         print(
            #             "... if you don't want to enable word prefix tree in decoding, set 'word-tree'=null")

        testsets = hyper_settings['data']['test']
        if isinstance(testsets, str):
            testsets = [testsets]
        f_scps = expandPath('s', testsets, cwd)
        checkExist('f', f_scps)

        try:
            import cat
        except ModuleNotFoundError:
            import sys
            sys.path.append(cwd)
        if hyper_settings['topo'] == 'rnnt':
            from cat.rnnt.decode import DecoderParser
            from cat.rnnt.decode import main as DecoderMain
        elif hyper_settings['topo'] == 'ctc':
            from cat.ctc.decode import DecoderParser
            from cat.ctc.decode import main as DecoderMain
        else:
            raise RuntimeError(f"Unknown topology: {hyper_settings['topo']}")

        for _set, scp in zip(testsets, f_scps):
            decode_settings['output_prefix'] = decode_out_prefix+f'_{_set}'
            decode_settings['input_scp'] = scp
            print(fmt.format(f"{scp} -> {decode_settings['output_prefix']}"))

            # FIXME: this canonot be spawned via mp_spawn, otherwise error would be raised
            DecoderMain(updateNamespaceFromDict(
                decode_settings, DecoderParser()))
            decode_settings['dist-url'] = f"tcp://localhost:{get_free_port()}"

        # compute wer/cer
        from wer import WERParser
        from wer import main as WERMain
        assert 'er' in inference_settings, fmt.format(
            "missing 'er' in hyper-p['inference']")
        err_settings = inference_settings['er']
        assert 'mode' in err_settings, fmt.format(
            "missing 'mode' in hyper-p['inference']['er']")

        f_texts = expandPath('t', testsets, cwd)
        checkExist('f', f_texts)

        if 'oracle' not in err_settings:
            err_settings['oracle'] = True
            print(fmt.format(f"set 'oracle' to True"))

        if 'stripif' not in err_settings:
            err_settings['stripid'] = True
            print(fmt.format(f"set 'stripid' to True"))

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
