"""Process of LM training
"""

from asr_process import (
    checkExist,
    resolve_sp_path,
    combineText,
    updateNamespaceFromDict,
    NNTrain,
    SentencePieceTrain,
    mp_spawn
)

import os
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

        assert 'sp' in hyper_settings, fmt.format(
            f"missing 'sp' in hyper-setting file {f_hyper_settings}")

        _, (spmodel, spvocab) = resolve_sp_path(hyper_settings['sp'])
        try:
            checkExist('f', [spmodel, spvocab])
            print(fmt.format(
                f"found existing sentencepiece model at {spmodel}, skipped training."))
        except FileNotFoundError:
            SentencePieceTrain(hyper_settings, f_hyper_settings, fmt)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        print("{0} {1} {0}".format("="*20, "Stage 2 Pickle data"))
        from transText2Bin import TextProcessingParser
        from transText2Bin import main as ProcessingMain
        fmt = "Stage 2  Pickle data: {}"

        assert 'data' in hyper_settings, fmt.format(
            f"missing 'data' in hyper-setting file {f_hyper_settings}")
        assert 'sp' in hyper_settings, fmt.format(
            f"missing 'sp' in hyper-setting file {f_hyper_settings}")

        _, (spmodel, _) = resolve_sp_path(hyper_settings['sp'])
        checkExist('f', spmodel)

        data_settings = hyper_settings['data']
        if 'text_processing' not in data_settings:
            processing_settings = {}
        else:
            processing_settings = data_settings['text_processing']
        if 'nj' not in processing_settings:
            processing_settings['nj'] = os.cpu_count()
            print(fmt.format(f"set 'nj' to {processing_settings['nj']}"))
        if 'spm' not in processing_settings:
            processing_settings['spm'] = spmodel
            print(fmt.format(f"set 'spm' to {spmodel}"))

        if 'lang' in hyper_settings['data']:
            # check if it's chinese-like languages
            iszh = ('zh' == hyper_settings['data']['lang'].split('-')[0])
        else:
            iszh = False

        if iszh:
            seperator = ''
        else:
            seperator = ' '

        pkldir = os.path.join(args.expdir, 'lmbin')
        os.makedirs(pkldir, exist_ok=True)
        for part in ['train', 'dev', 'test']:
            assert part in data_settings, fmt.format(
                f"missing '{part}' in hyper-p['data']")
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
        print("{0} {1} {0}".format("="*20, "Stage 3 NN training"))
        fmt = "Stage 3  NN training: {}"
        try:
            import cat
        except ModuleNotFoundError:
            import sys
            sys.path.append(cwd)
        from cat.lm.train import LMParser
        from cat.lm.train import main as LMMain

        NNTrain(args, hyper_settings, f_hyper_settings, os.path.join(
            args.expdir, 'lmbin/{}.pkl'), LMParser(), LMMain, fmt)

    ############ Stage 4  Evaluating ############
    if s_beg <= 4 and s_end >= 4:
        print("{0} {1} {0}".format("="*20, "Stage 4 Evaluating"))
        fmt = "Stage 4  Evaluating: {}"
        try:
            import cat
        except ModuleNotFoundError:
            import sys
            sys.path.append(cwd)
        from cat.lm.train import LMParser
        from cat.lm.train import main as LMMain
        if 'resume' not in hyper_settings['train']:
            hyper_settings['train']['resume'] = os.path.join(
                args.expdir, 'checks/bestckpt.pt')
        print(fmt.format(
            f"set 'resume' to {hyper_settings['train']['resume']}"))
        if 'eval' not in hyper_settings['train']:
            hyper_settings['train']['eval'] = os.path.join(
                args.expdir, 'lmbin/test.pkl')
            print(fmt.format(
                f"set 'eval' to {hyper_settings['train']['eval']}"))
        NNTrain(args, hyper_settings, f_hyper_settings, os.path.join(
            args.expdir, 'lmbin/{}.pkl'), LMParser(), LMMain, fmt)
