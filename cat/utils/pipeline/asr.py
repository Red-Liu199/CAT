"""Ported from run_rnnt.sh, rewrote with python

Uage:
    python utils/pipeline/asr.py
"""

try:
    from _constants import (
        F_DATAINFO,
        F_NN_CONFIG,
        F_HYPER_CONFIG,
        D_CHECKPOINT,
        D_INFER
    )
except ModuleNotFoundError:
    from cat.shared._constants import (
        F_DATAINFO,
        F_NN_CONFIG,
        F_HYPER_CONFIG,
        D_CHECKPOINT,
        D_INFER
    )


import os
import sys
import json
import uuid
import pickle
import argparse
from typing import *
from multiprocessing import Process

# NOTE: the escape sequences do not support nested using


class tcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _fmtstr(s: str, ct: Literal):
    assert ct != tcolors.ENDC
    return f"{ct}{s}{tcolors.ENDC}"


def udl(s: str):
    return _fmtstr(s, tcolors.UNDERLINE)


def fmtstr_warn(prompt: str, func: Optional[Callable] = None):
    if func is None:
        return f"{_fmtstr('WARNING:', tcolors.WARNING)} {prompt}"
    else:
        return f"{_fmtstr('WARNING:', tcolors.WARNING)} {func.__name__}() {prompt}"


def fmtstr_error(prompt: str, func: Optional[Callable] = None):
    if func is None:
        return f"{_fmtstr('ERROR:', tcolors.FAIL)} {prompt}"
    else:
        return f"{_fmtstr('ERROR:', tcolors.FAIL)} {func.__name__}() {prompt}"


def fmtstr_header(prompt: str):
    return "{0} {1} {0}".format('='*20, _fmtstr(prompt, tcolors.BOLD))


def fmtstr_missing(property_name: str, field: Optional[Union[str, Iterable[str]]] = None, raiseerror: bool = True):
    if raiseerror:
        formatter = fmtstr_error
    else:
        formatter = fmtstr_warn

    if isinstance(field, str):
        field = [field]
    if field is None:
        return formatter(f"missing '{property_name}'")
    else:
        return formatter(f"missing '{property_name}' in {udl(':'.join(field))}")


def fmtstr_set(property_name: str, value: str, isPath: bool = True):
    if isPath:
        value = udl(value)
    return f"set '{property_name}' -> {value}"


def initial_datainfo():
    if not os.path.isfile(F_DATAINFO):
        from cat.utils.data import resolvedata
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


def readjson(file: str) -> dict:
    checkExist('f', file)
    with open(file, 'r') as fi:
        data = json.load(fi)
    return data


def dumpjson(obj: dict, target: str):
    assert os.access(os.path.dirname(target),
                     os.W_OK), f"{target} is not writable."
    with open(target, 'w') as fo:
        json.dump(obj, fo, indent=4)


class TextUtterances:
    """Read files with uid and sort the utterances in order by uid."""

    def __init__(self, files: Union[str, List[str]]) -> None:
        if isinstance(files, str):
            files = [files]

        checkExist('f', files)
        # [(uid, seek, file_id), ...]
        self._seeks = []    # type: List[Tuple[str, int, int]]
        self._files = sorted(files)

        for idf, f in enumerate(files):
            with open(f, 'r') as fi:
                while True:
                    loc = fi.tell()
                    line = fi.readline()
                    if line == '':
                        break
                    uid = line.split(maxsplit=1)[0]
                    self._seeks.append(
                        (uid, loc, idf)
                    )
        self._seeks = sorted(self._seeks, key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self._seeks)

    def __getitem__(self, index: int):
        return self._seeks[index]

    def __iter__(self):
        opened = {}
        for uid, loc, idf in self._seeks:
            if idf not in opened:
                opened[idf] = open(self._files[idf], 'r')

            opened[idf].seek(loc)
            cont = opened[idf].readline().split(maxsplit=1)
            if len(cont) == 1:
                yield (uid, '')
            else:
                yield (uid, cont[1])

        for f in opened.values():
            f.close()
        return


def pack_data(
        f_scps: Union[List[str], str],
        f_labels: Union[List[str], str],
        f_out: str,
        tokenizer,
        filter: Optional[str] = None):
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
        sys.stderr.write(
            fmtstr_warn(
                f"file exist: {udl(f_out)}, "
                "rm it if you want to update the data.\n",
                pack_data
            )
        )
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
        assert ':' in filter, fmtstr_error(
            f"invalid filter format {filter}", pack_data)
        l_bound, u_bound = (i for i in filter.split(':'))
        if l_bound != '':
            l_min = int(l_bound)
        if u_bound != '':
            l_max = int(u_bound)

    # Read label files and scp files.
    twrapper_label = TextUtterances(f_labels)
    twrapper_scp = TextUtterances(f_scps)
    assert len(twrapper_scp) == len(twrapper_label), fmtstr_error(
        "f_scp and f_label should match on the # of lines, "
        f"instead {len(twrapper_scp)} != {len(twrapper_label)}",
        pack_data
    )

    f_opened = {}
    cnt_frames = 0
    linfo = np.empty(len(twrapper_scp), dtype=np.int64)
    uids = []
    arks = []
    labels = []
    cnt = 0
    for (uid, lb), (uid1, ark) in tqdm(zip(twrapper_label, twrapper_scp), total=len(twrapper_scp), leave=False):
        assert uid == uid1, f"UID in label and scp files mismatch: {uid} != {uid1}"
        if lb == '':
            fmtstr_warn(f"skip empty utt: {uid}", pack_data)
            continue

        mat = kaldiio.load_mat(
            ark, fd_dict=f_opened)   # type:np.ndarray
        if mat.shape[0] < l_min or mat.shape[0] > l_max:
            continue

        labels.append(
            np.asarray(
                tokenizer.encode(lb),
                dtype=np.int64
            )
        )
        linfo[cnt] = mat.shape[0]
        uids.append(uid)
        arks.append(ark)
        cnt_frames += mat.shape[0]
        cnt += 1

    for f in f_opened.values():
        f.close()

    # in order to store labels in a ndarray,
    # first I pad all labels to the max length with -1 (this won't take many memory since labels are short compared to frames)
    # then store the length in the last place, such as
    # [0 1 2 3] -> [0 1 2 3 -1 -1 4]
    # then we can access the data via array[:array[-1]]
    cnt_tokens = sum(x.shape[0] for x in labels)
    max_len_label = max(x.shape[0] for x in labels)
    labels = np.array([
        np.concatenate((
            _x,
            np.array([-1]*(max_len_label-_x.shape[0]) + [_x.shape[0]])
        ))
        for _x in labels
    ])

    with open(f_out, 'wb') as fo:
        pickle.dump({
            'label': labels,
            'linfo': linfo[:cnt],
            'arkname': np.array(arks),
            'key': np.array(uids)
        }, fo)

    cntrm = len(twrapper_scp) - cnt
    if cntrm > 0:
        print(f"pack_data(): remove {cntrm} unqualified sequences.")
    print(
        f"# of frames: {cnt_frames} | tokens: {cnt_tokens} | seqs: {cnt}")


def checkExist(f_type: Literal['d', 'f'], f_list: Union[str, List[str]]):
    """Check whether directory/file exist and raise error if it doesn't.
    """
    if f_type == 'd':
        check = os.path.isdir
    elif f_type == 'f':
        check = os.path.isfile
    else:
        raise RuntimeError(fmtstr_error(
            f"unknown f_type: {f_type}, expected one of ['d', 'f']",
            checkExist
        ))

    if isinstance(f_list, str):
        f_list = [f_list]
    assert len(f_list) > 0, fmtstr_error(
        f"expect the file/dir list to have at least one element, but found empty.",
        checkExist
    )

    hints = {'d': 'Directory', 'f': 'File'}
    not_founds = []

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
        prompt: str = '{}\n'):

    checkExist('f', f_hyper_p)
    settings = readjson(f_hyper_p)
    assert 'train' in settings, fmtstr_missing(
        'train', udl(f_hyper_p)
    )
    assert 'bin' in settings['train'], fmtstr_missing(
        'bin', (udl(f_hyper_p), 'train')
    )
    assert 'option' in settings['train'], fmtstr_missing(
        'option', (udl(f_hyper_p), 'train')
    )

    f_nnconfig = os.path.join(working_dir, F_NN_CONFIG)
    if 'tokenizer' not in settings:
        sys.stderr.write(
            fmtstr_missing('tokenizer', raiseerror=False) + '\n' +
            fmtstr_warn(
                f"you have to ensure the 'num_classes' in {udl(f_nnconfig)} is correct.\n",
                train_nn_model
            )
        )
    else:
        if '|V|' in settings['tokenizer']:
            vocab_size = settings['tokenizer']['|V|']
        else:
            import cat.shared.tokenizer as tknz
            checkExist('f', settings['tokenizer']['file'])
            vocab_size = tknz.load(settings['tokenizer']['file']).vocab_size

        checkExist('f', f_nnconfig)
        nnconfig = readjson(f_nnconfig)
        # recursively search for 'num_classes'
        recursive_rpl(nnconfig, 'num_classes', vocab_size)
        dumpjson(nnconfig, f_nnconfig)

    train_options = settings['train']['option']
    if 'trset' not in train_options:
        train_data = fmt_data.format('train')
        checkExist('f', train_data)
        train_options['trset'] = train_data
        sys.stdout.write(prompt.format(fmtstr_set('trset', train_data)))
    if 'devset' not in train_options:
        dev_data = fmt_data.format('dev')
        checkExist('f', dev_data)
        train_options['devset'] = dev_data
        sys.stdout.write(prompt.format(fmtstr_set('devset', dev_data)))
    if 'dir' not in train_options:
        train_options['dir'] = working_dir
        sys.stdout.write(prompt.format(fmtstr_set('dir', working_dir)))
    if 'dist-url' not in train_options:
        train_options['dist-url'] = f"tcp://localhost:{get_free_port()}"
        sys.stdout.write(prompt.format(fmtstr_set(
            'dist-url', train_options['dist-url'], False)))

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

    datainfo = readjson(F_DATAINFO)

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
        text_withid) > 0, fmtstr_error(
            f"dataset '{datasets}' not found.",
            combine_text
    )

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
                    contents = line.split(maxsplit=1)
                    if len(contents) == 1:
                        continue
                    fo.write(contents[1])
    return f_out


def train_tokenizer(f_hyper: str):
    def update_conf(_tok, path):
        # store some info about the tokenizer to the file
        cfg_hyper = readjson(f_hyper)
        cfg_hyper['tokenizer']['|V|'] = _tok.vocab_size
        cfg_hyper['tokenizer']['file'] = path
        dumpjson(cfg_hyper, f_hyper)

    checkExist('f', f_hyper)
    cfg_hyper = readjson(f_hyper)

    import cat.shared.tokenizer as tknz

    assert 'tokenizer' in cfg_hyper, fmtstr_error(
        "'tokenizer' is not configured.", train_tokenizer)
    if 'file' not in cfg_hyper['tokenizer']:
        f_tokenizer = os.path.join(
            os.path.dirname(f_hyper), 'tokenizer.tknz')
        sys.stdout.write(
            "train_tokenizer(): " +
            fmtstr_set(
                'tokenizer:file', f_tokenizer)+'\n'
        )
    else:
        f_tokenizer = cfg_hyper['tokenizer']['file']
        if os.path.isfile(f_tokenizer):
            update_conf(tknz.load(f_tokenizer), f_tokenizer)
            sys.stderr.write(
                fmtstr_warn(
                    f"['tokenizer']['file'] exists: {udl(f_tokenizer)}\n"
                    "... skip tokenizer training. If you want to do tokenizer training anyway,\n"
                    "... remove the ['tokenizer']['file'] in setting\n"
                    f"... or remove the file:{udl(f_tokenizer)} then re-run the script.\n",
                    train_tokenizer
                )
            )
            return

    assert 'data' in cfg_hyper, fmtstr_error(
        "data is not configured.", train_tokenizer)
    assert 'train' in cfg_hyper['data'], fmtstr_error(
        "data:train is not configured.", train_tokenizer)
    assert os.access(os.path.dirname(f_tokenizer), os.W_OK), \
        f"tokenizer:file is not writable: '{udl(cfg_hyper['tokenizer']['file'])}'"

    sys.stderr.write(
        fmtstr_warn(
            "for Asian languages, "
            "it's your duty to remove the segment spaces up to your requirement.\n",
            train_tokenizer
        )
    )

    f_text = None
    # combine the transcripts and remove the ids if needed.
    if 'option-train' in cfg_hyper['tokenizer']:
        if 'f_text' not in cfg_hyper['tokenizer']['option-train']:
            f_text = combine_text(cfg_hyper['data']['train'])
            cfg_hyper['tokenizer']['option-train']['f_text'] = f_text

    tokenizer = tknz.initialize(cfg_hyper['tokenizer'])
    tknz.save(tokenizer, f_tokenizer)
    if f_text is not None:
        os.remove(f_text)

    update_conf(tokenizer, f_tokenizer)


def model_average(
        setting: dict,
        checkdir: str,
        returnifexist: bool = False) -> Tuple[str, str]:
    """Do model averaging according to given setting, return the averaged model path."""

    assert 'mode' in setting, fmtstr_error(
        "'mode' not specified.", model_average)
    assert 'num' in setting, fmtstr_error(
        "'num' not specified.", model_average)
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
            fmtstr_warn(
                "git command not found, skip logging commit.\n",
                log_commit
            )
        )
    else:
        process = subprocess.run(
            "git log -n 1 --pretty=format:\"%H\"", shell=True, check=True, stdout=subprocess.PIPE)

        orin_settings = readjson(f_hyper)
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
    hyper_cfg = readjson(f_hyper)
    if "env" in hyper_cfg:
        for k, v in hyper_cfg["env"].items():
            os.environ[k] = v
    if 'commit' not in hyper_cfg:
        log_commit(f_hyper)

    # setting visible gpus before loading cat/torch
    if args.ngpu > -1:
        set_visible_gpus(args.ngpu)

    from cat.shared import tokenizer as tknz

    initial_datainfo()
    datainfo = readjson(F_DATAINFO)

    ############ Stage 1  Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silent:
            print(fmtstr_header("Stage 1 Tokenizer training"))
            fmt = _fmtstr(_fmtstr("Tokenizer training: ",
                          tcolors.BOLD), tcolors.OKCYAN) + "{}\n"
        else:
            fmt = ''

        hyper_cfg = readjson(f_hyper)
        if 'tokenizer' not in hyper_cfg:
            sys.stderr.write(
                fmtstr_missing('tokenizer', raiseerror=False) +
                ", skip tokenizer training.\n"
            )
        else:
            train_tokenizer(f_hyper)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silent:
            print(fmtstr_header("Stage 2 Pickle data"))
            fmt = _fmtstr(_fmtstr("Pickle data: ",
                          tcolors.BOLD), tcolors.OKCYAN) + "{}\n"
        else:
            fmt = ''

        hyper_cfg = readjson(f_hyper)
        assert 'data' in hyper_cfg, fmtstr_missing('data', udl(f_hyper))
        # load tokenizer from file
        assert 'tokenizer' in hyper_cfg, fmtstr_missing(
            'tokenizer', udl(f_hyper))
        assert 'file' in hyper_cfg['tokenizer'], fmtstr_missing(
            'file', (udl(f_hyper), 'tokenizer'))

        f_tokenizer = hyper_cfg['tokenizer']['file']
        checkExist('f', f_tokenizer)
        tokenizer = tknz.load(f_tokenizer)

        data_settings = hyper_cfg['data']
        if 'filter' not in data_settings:
            data_settings['filter'] = None

        d_pkl = os.path.join(working_dir, 'pkl')
        os.makedirs(d_pkl, exist_ok=True)
        for dataset in ['train', 'dev', 'test']:
            if dataset not in data_settings:
                sys.stderr.write(
                    fmtstr_missing(dataset, 'data', raiseerror=False) +
                    ", skip.\n"
                )
                continue

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
                filter=filter, tokenizer=tokenizer
            )
            del f_data

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silent:
            print(fmtstr_header("Stage 3 NN training"))
            fmt = _fmtstr(_fmtstr("NN training: ",
                          tcolors.BOLD), tcolors.OKCYAN) + "{}\n"
        else:
            fmt = ''

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
            print(fmtstr_header("Stage 4 Decode"))
            fmt = _fmtstr(_fmtstr("Decode: ",
                          tcolors.BOLD), tcolors.OKCYAN) + "{}\n"
        else:
            fmt = ''

        hyper_cfg = readjson(f_hyper)
        assert 'inference' in hyper_cfg, fmtstr_missing(
            'inference', udl(f_hyper))

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
            assert 'bin' in cfg_infr['infer'], fmtstr_missing(
                'bin', (udl(f_hyper), 'inference', 'infer'))
            assert 'option' in cfg_infr['infer'], fmtstr_missing(
                'option', (udl(f_hyper), 'inference', 'infer'))

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
                        fmtstr_missing(
                            'resume', ('inference', 'infer', 'option'), False
                        ) +
                        "\n    ... would causing non-initialized evaluation.\n"
                    )
                else:
                    # there's no way the output of model_average() is an invalid path
                    # ... so here we could skip the checkExist()
                    # update config to file
                    _hyper = readjson(f_hyper)
                    _hyper['inference']['infer']['option']['resume'] = checkpoint
                    dumpjson(_hyper, f_hyper)
                    infr_option['resume'] = checkpoint
                    sys.stdout.write(fmt.format(fmtstr_set(
                        'inference:infer:option:resume',
                        checkpoint
                    )))
            else:
                sys.stdout.write(fmt.format(
                    "setting 'resume' in inference:infer:option "
                    "would ignore the inference:avgmodel settings."
                ))
                checkpoint = infr_option['resume']
                checkExist('f', checkpoint)

            if 'config' not in infr_option:
                infr_option['config'] = os.path.join(working_dir, F_NN_CONFIG)
                checkExist('f', infr_option['config'])

            intfname = cfg_infr['infer']['bin']
            # check tokenizer
            if intfname != "cat.ctc.cal_logit":
                if 'tokenizer' not in infr_option:
                    assert hyper_cfg.get('tokenizer', {}).get('file', None) is not None, \
                        (
                        "\nyou should set at least one of:\n"
                        f"1. set tokenizer:file ;\n"
                        f"2. set inference:infer:option:tokenizer \n"
                    )
                    infr_option['tokenizer'] = hyper_cfg['tokenizer']['file']

            ignore_field_data = False
            os.makedirs(f"{working_dir}/decode", exist_ok=True)
            if intfname == 'cat.ctc.cal_logit':
                if 'input_scp' in infr_option:
                    ignore_field_data = True

                if 'output_dir' not in infr_option:
                    assert not ignore_field_data
                    infr_option['output_dir'] = os.path.join(
                        working_dir, D_INFER+'/{}')
                    sys.stdout.write(fmt.format(fmtstr_set(
                        'inference:infer:option:output_dir',
                        infr_option['output_dir']
                    )))
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
                    if 'unified' in infr_option and infr_option['unified']:
                        prefix += f"_streaming_{infr_option.get('streaming', 'false')}"
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
                sys.stderr.write(fmtstr_warn(
                    f"interface '{intfname}' only support handcrafted execution.\n"
                ))

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
                assert 'data' in hyper_cfg, fmtstr_missing(
                    'data', udl(f_hyper))
                assert 'test' in hyper_cfg['data'], fmtstr_missing(
                    'test', (udl(f_hyper), 'data'))

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
                            sys.stdout.write(fmt.format(f"{_set}: " + fmtstr_set(
                                k, running_option[k]
                            )))
                    running_option['input_scp'] = scp
                    if intfname in ['cat.ctc.decode', 'cat.rnnt.decode']:
                        if os.path.isfile(running_option['output_prefix']):
                            sys.stderr.write(fmtstr_warn(
                                f"{udl(running_option['output_prefix'])} exists, skip.\n"
                            ))
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
                assert 'gt' in err_option, fmtstr_missing(
                    'gt', (udl(f_hyper), 'inference', 'er'))
                wercal.main(get_args(
                    err_option,
                    wercal._parser(),
                    [err_option['gt'], err_option['hy']]
                ))
