"""Process of LM training
"""

from asr_process import (
    checkExist,
    combineText,
    updateNamespaceFromDict,
    TrainNNModel,
    TrainTokenizer,
    mp_spawn,
    readfromjson
)

import os
import sys
import json
import argparse


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

    ############ Stage 1 Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 1 Tokenizer training"))
        fmt = "# Tokenizer trainin # {}\n" if not args.silient else ""
        hyper_settings = readfromjson(f_hyper_settings)
        if 'tokenizer' not in hyper_settings:
            sys.stderr.write(
                f"warning: missing 'tokenizer' in hyper-setting, skip tokenizer training.")
        else:
            TrainTokenizer(f_hyper_settings)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 2 Pickle data"))
        fmt = "# Pickle data # {}\n" if not args.silient else ""
        from transText2Bin import TextProcessingParser
        from transText2Bin import main as ProcessingMain

        hyper_settings = readfromjson(f_hyper_settings)
        assert 'data' in hyper_settings, f"missing 'data' in hyper-setting file {f_hyper_settings}"

        data_settings = hyper_settings['data']
        if 'text_processing' not in data_settings:
            processing_settings = {}
        else:
            processing_settings = data_settings['text_processing']
        if 'nj' not in processing_settings:
            processing_settings['nj'] = os.cpu_count()
            sys.stdout.write(fmt.format(
                f"set 'nj' to {processing_settings['nj']}"))

        if 'raw-tokenizer' in processing_settings and processing_settings['raw-tokenizer']:
            pass
        elif 'tokenizer' not in processing_settings:
            assert 'tokenizer' in hyper_settings, (
                "\n"
                "At least one of these options is required:\n"
                "1. set 'raw-tokenizer' if the text corpus is tokenized;\n"
                "2. specify 'tokenizer' in ['data']['text_processing'];\n"
                f"3. setup 'tokenizer' and ['tokenizer']['location'] in {f_hyper_settings}\n")
            processing_settings['tokenizer'] = hyper_settings['tokenizer']['location']
            sys.stdout.write(
                f"set ['text_processing']['tokenizer']='{processing_settings['tokenizer']}'")

        if 'lang' in hyper_settings['data']:
            # check if it's chinese-like languages
            iszh = ('zh' == hyper_settings['data']['lang'].split('-')[0])
        else:
            iszh = False
        seperator = '' if iszh else ' '

        pkldir = os.path.join(args.expdir, 'lmbin')
        os.makedirs(pkldir, exist_ok=True)
        for part in ['train', 'dev', 'test']:
            if part not in data_settings:
                sys.stderr.write(
                    f"warining: missing '{part}' in hyper-p['data'], skip\n")
                continue
            part_text = combineText(data_settings[part], seperator=seperator)
            if part != 'train':
                setting = processing_settings.copy()
                if 'concat' in setting:
                    del setting['concat']
                if 'truncate' in setting:
                    del setting['truncate']
            else:
                setting = processing_settings

            f_pkl = os.path.join(pkldir, part+'.pkl')
            mp_spawn(ProcessingMain, updateNamespaceFromDict(
                setting, TextProcessingParser(), [part_text, f_pkl]))

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 3 NN training"))
        fmt = "# NN training # {}\n" if not args.silient else ""
        try:
            import cat
        except ModuleNotFoundError:
            import sys
            sys.path.append(cwd)
        from cat.lm.train import LMParser
        from cat.lm.train import main as LMMain

        hyper_settings = readfromjson(f_hyper_settings)
        TrainNNModel(args, hyper_settings, f_hyper_settings, os.path.join(
            args.expdir, 'lmbin/{}.pkl'), LMParser(), LMMain, fmt)

    ############ Stage 4  Evaluate ############
    if s_beg <= 4 and s_end >= 4:
        if not args.silient:
            print("{0} {1} {0}".format("="*20, "Stage 4 Evaluate"))
        fmt = "# Evaluate # {}\n" if not args.silient else ""
        try:
            import cat
        except ModuleNotFoundError:
            import sys
            sys.path.append(cwd)
        from cat.lm.train import LMParser
        from cat.lm.train import main as LMMain
        hyper_settings = readfromjson(f_hyper_settings)
        if 'resume' not in hyper_settings['train']:
            hyper_settings['train']['resume'] = os.path.join(
                args.expdir, 'checks/bestckpt.pt')
        sys.stdout.write(fmt.format(
            f"set 'resume' to {hyper_settings['train']['resume']}"))
        if 'eval' not in hyper_settings['train']:
            hyper_settings['train']['eval'] = os.path.join(
                args.expdir, 'lmbin/test.pkl')
            sys.stdout.write(fmt.format(
                f"set 'eval' to {hyper_settings['train']['eval']}"))
        TrainNNModel(args, hyper_settings, f_hyper_settings, os.path.join(
            args.expdir, 'lmbin/{}.pkl'), LMParser(), LMMain, fmt)
