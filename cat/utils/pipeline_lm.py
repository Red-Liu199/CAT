"""Process of LM training
"""

from pipeline_asr import *

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
    checkExist('d', args.expdir)
    f_hyper_settings = os.path.join(args.expdir, 'hyper-p.json')
    checkExist('f', f_hyper_settings)
    if args.ngpu > -1:
        set_visible_gpus(args.ngpu)
    initial_datainfo()

    ############ Stage 1 Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 1 Tokenizer training"))
        fmt = "# Tokenizer trainin # {}\n" if not args.silent else ""
        hyper_settings = readfromjson(f_hyper_settings)
        if 'tokenizer' not in hyper_settings:
            sys.stderr.write(
                f"warning: missing 'tokenizer' in hyper-setting, skip tokenizer training.\n")
        else:
            TrainTokenizer(f_hyper_settings)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 2 Pickle data"))
        fmt = "# Pickle data # {}\n" if not args.silent else ""
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
            processing_settings['nj'] = max(1, os.cpu_count() // 2)
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
            sys.stdout.write(fmt.format(
                f"set ['text_processing']['tokenizer']='{processing_settings['tokenizer']}'"))

        pkldir = os.path.join(args.expdir, 'lmbin')
        os.makedirs(pkldir, exist_ok=True)
        # 'train' and 'dev' datasets would be merged into ones,
        # 'test' datasets would be processed individually in stage 4s
        for part in ['train', 'dev']:
            if part not in data_settings:
                sys.stderr.write(
                    f"warining: missing '{part}' in field:['data'], skip\n")
                continue
            part_text = combineText(data_settings[part])
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
            os.remove(part_text)

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 3 NN training"))
        fmt = "# NN training # {}\n" if not args.silent else ""
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
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 4 Evaluate"))
        fmt = "# Evaluate # {}\n" if not args.silent else ""

        hyper_settings = readfromjson(f_hyper_settings)
        assert 'data' in hyper_settings, f"missing 'data': {f_hyper_settings}"
        if 'test' not in hyper_settings['data']:
            sys.stderr.write(
                "missing 'test' in field:['data'], skip evaluation.\n")
            sys.exit(0)

        if 'inference' not in hyper_settings:
            hyper_settings['inference'] = {}

        infer_setting = hyper_settings['inference']
        if 'avgmodel' in infer_setting:
            # do model averaging
            checkpoint, _ = model_average(
                setting=infer_setting['avgmodel'],
                checkdir=os.path.join(args.expdir, 'checks'),
                returnifexist=True)
        else:
            checkpoint = None

        # compute ppl
        if 'ppl' not in infer_setting:
            infer_setting['ppl'] = {}

        if 'tokenizer' not in infer_setting['ppl']:
            assert 'tokenizer' in hyper_settings, f"missing 'tokenizer' in field:"
            assert 'location' in hyper_settings['tokenizer'], "missing 'tokenizer' in field:['tokenizer']"
            f_tokenizer = hyper_settings['tokenizer']['location']
            checkExist('f', f_tokenizer)
            infer_setting['ppl']['tokenizer'] = f_tokenizer

        if 'resume' in infer_setting['ppl']:
            sys.stdout.write(fmt.format(
                "setting 'resume' in field:['inference']['ppl'] would overwrite field:['inference']['avgmodel']"))
        elif checkpoint is not None:
            sys.stderr.write(fmt.format(f"set 'resume' to {checkpoint}"))
            infer_setting['ppl']['resume'] = checkpoint

        text_local, text_resolved = resolve_in_priority(
            hyper_settings['data']['test'])

        f_normalized = text_local + \
            [combineText(t_r, t_r+'.tmp') for t_r in text_resolved]
        os.system(" ".join([
            sys.executable,
            "-m cat.lm.ppl_compute",
            os.path.join(args.expdir, 'config.json'),
            "-e {}".format(' '.join(f_normalized)),
            " ".join(f"--{opt} {val}" for opt,
                     val in infer_setting['ppl'].items())
        ]))
