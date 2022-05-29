"""Process of LM training
"""

from asr import *

import os
import sys
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
    from cat.shared._constants import (
        F_NN_CONFIG,
        F_HYPER_CONFIG
    )

    cwd = os.getcwd()
    working_dir = args.expdir
    checkExist('d', working_dir)
    f_hyper = os.path.join(working_dir, F_HYPER_CONFIG)
    checkExist('f', f_hyper)
    hyper_cfg = readfromjson(f_hyper)
    if "env" in hyper_cfg:
        for k, v in hyper_cfg["env"].items():
            os.environ[k] = v

    if 'commit' not in hyper_cfg:
        log_commit(f_hyper)

    if args.ngpu > -1:
        set_visible_gpus(args.ngpu)
    initial_datainfo()

    ############ Stage 1 Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 1 Tokenizer training"))
        fmt = "# Tokenizer trainin # {}\n" if not args.silent else ""
        hyper_cfg = readfromjson(f_hyper)
        if 'tokenizer' not in hyper_cfg:
            sys.stderr.write(
                f"warning: missing 'tokenizer' in hyper-setting, skip tokenizer training.\n")
        else:
            train_tokenizer(f_hyper)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 2 Pickle data"))
        fmt = "# Pickle data # {}\n" if not args.silent else ""
        from cat.utils.data import pack_corpus as t2b

        hyper_cfg = readfromjson(f_hyper)
        assert 'data' in hyper_cfg, f"missing 'data' in hyper-setting file {f_hyper}"

        data_settings = hyper_cfg['data']
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
            assert 'tokenizer' in hyper_cfg, (
                "\n"
                "At least one of these options is required:\n"
                "1. set 'raw-tokenizer' if the text corpus is tokenized;\n"
                "2. specify 'tokenizer' in ['data']['text_processing'];\n"
                f"3. setup 'tokenizer' and ['tokenizer']['location'] in {f_hyper}\n")
            processing_settings['tokenizer'] = hyper_cfg['tokenizer']['location']

        pkldir = os.path.join(working_dir, 'lmbin')
        os.makedirs(pkldir, exist_ok=True)
        # 'train' and 'dev' datasets would be merged into ones,
        # 'test' datasets would be processed individually in stage 4s
        for part in ['train', 'dev']:
            if part not in data_settings:
                sys.stderr.write(
                    f"warining: missing '{part}' in field:data, skip\n")
                continue
            part_text = combine_text(data_settings[part])
            if part != 'train':
                setting = processing_settings.copy()
                if 'concat' in setting:
                    del setting['concat']
                if 'truncate' in setting:
                    del setting['truncate']
            else:
                setting = processing_settings

            f_pkl = os.path.join(pkldir, part+'.pkl')
            if os.path.isfile(f_pkl):
                sys.stderr.write(f"warning: {f_pkl} exists, skip.\n")
            else:
                mp_spawn(t2b.main, get_args(
                    setting, t2b.TextProcessingParser(), [part_text, f_pkl]))
            os.remove(part_text)

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 3 NN training"))
        fmt = "# NN training # {}\n" if not args.silent else ""

        train_nn_model(
            working_dir,
            f_hyper,
            f'{working_dir}'+'/lmbin/{}.pkl',
            fmt
        )

    ############ Stage 4  Evaluate ############
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

        from cat.shared._constants import (
            D_CHECKPOINT
        )
        if not args.silent:
            print("{0} {1} {0}".format("="*20, "Stage 4 Evaluate"))
        fmt = "# Evaluate # {}\n" if not args.silent else ""

        hyper_cfg = readfromjson(f_hyper)
        assert 'inference' in hyper_cfg, f"missing 'inference' at {f_hyper}"

        cfg_infr = hyper_cfg['inference']
        checkdir = os.path.join(working_dir, D_CHECKPOINT)
        # do model averaging
        if 'avgmodel' in cfg_infr:
            checkpoint = model_average(
                setting=cfg_infr['avgmodel'],
                checkdir=checkdir,
                returnifexist=True
            )[0]
        else:
            checkpoint = None

        assert 'data' in hyper_cfg, f"missing 'data': {f_hyper}"
        if 'test' not in hyper_cfg['data']:
            sys.stderr.write(
                "missing 'test' in field:data, skip evaluation.\n")
            sys.exit(0)

        if 'inference' not in hyper_cfg:
            hyper_cfg['inference'] = {}

        infer_setting = hyper_cfg['inference']
        if 'avgmodel' in infer_setting:
            # do model averaging
            checkpoint = model_average(
                setting=infer_setting['avgmodel'],
                checkdir=checkdir,
                returnifexist=True
            )[0]
        else:
            checkpoint = None

        if 'infer' not in infer_setting:
            cfg_infr['infer'] = {}

        # defaultly compute ppl
        if 'bin' not in cfg_infr['infer']:
            cfg_infr['infer']['bin'] = 'cat.lm.ppl_compute'
        if 'option' not in cfg_infr['infer']:
            cfg_infr['infer']['option'] = {}

        infr_option = cfg_infr['infer']['option']
        intfname = cfg_infr['infer']['bin']

        # check config
        if 'config' not in infr_option:
            infr_option['config'] = os.path.join(working_dir, F_NN_CONFIG)
            checkExist('f', infr_option['config'])
        # check tokenizer
        if 'tokenizer' not in infr_option:
            assert hyper_cfg.get('tokenizer', {}).get('location', None) is not None, \
                (
                "\nyou should set at least one of:\n"
                f"1. set tokenizer:location ;\n"
                f"2. set inference:infer:option:tokenizer \n"
            )
            infr_option['tokenizer'] = hyper_cfg['tokenizer']['location']

        import importlib
        interface = importlib.import_module(intfname)
        # since the lm is likely to be n-gram one. checkpoint is None is ok.
        if 'resume' not in infr_option and checkpoint is not None:
            infr_option['resume'] = checkpoint
            sys.stdout.write(fmt.format(
                f"set inference:infer:option:resume -> {checkpoint}"
            ))

        if intfname == 'cat.lm.ppl_compute':
            # we need to remove the uid in the transcript text
            # but for text resovled from local path, we assume it's raw text w/o uid.
            text_local, _ = resolve_in_priority(hyper_cfg['data']['test'])
            # use combine_text() to remove the uid
            text_trans = [
                combine_text(t_r, f"/tmp/{t_r}.tmp")
                for t_r in set(hyper_cfg['data']['test']) - set(text_local)
            ]

            infr_option['evaluate'] = text_local+text_trans
            interface.main(get_args(
                infr_option,
                interface._parser(),
                [infr_option['config']]
            ))

            for t_r in text_trans:
                os.remove(t_r)
        elif intfname == 'cat.lm.rescore':
            # if the input is a format string like rnnt-16_{}.nbest
            # we use it to format the test sets.
            assert 'nbestlist' in infr_option, \
                f"missing 'nbestlist' in inference:infer:option"

            if infr_option.get('output', None) is None:
                suffix = os.path.basename(
                    infr_option['nbestlist']).removesuffix('.nbest')
                a = infr_option.get('alpha', 0)
                b = infr_option.get('beta', 0)
                if a != 0 or b != 0:
                    suffix = f"lm-a{a}b{b}_{suffix}"

                infr_option['output'] = os.path.join(
                    working_dir,
                    f"rescore/{suffix}"
                )
                os.makedirs(os.path.dirname(
                    infr_option['output']), exist_ok=True)
                if '{}' not in suffix:
                    sys.stdout.write(fmt.format(
                        f"set inference:infer:option:output -> {infr_option['output']}"
                    ))

            if '{}' in infr_option['nbestlist']:
                informatstr = True
                assert '{}' in infr_option['output'], \
                    f"you set 'nbestlist' as format string: {infr_option['nbestlist']}\n" \
                    f"... but the 'output' is not: {infr_option['output']}"

                if infr_option.get('save_lm_nbest', None) is not None and '{}' not in infr_option['save_lm_nbest']:
                    sys.stderr.write(
                        "Error:\n"
                        f"    you set 'nbestlist' as format string: {infr_option['nbestlist']}\n"
                        f"    ... but the 'save_lm_nbest' is not: {infr_option['save_lm_nbest']}\n"
                    )
                    sys.exit(1)
            else:
                informatstr = False

            if informatstr:
                assert 'data' in hyper_cfg, f"missing 'data' at {f_hyper}"
                assert 'test' in hyper_cfg[
                    'data'], f"missing 'test' in data: at {f_hyper}"

                testsets = hyper_cfg['data']['test']
                if isinstance(testsets, str):
                    testsets = [testsets]

                running_option = infr_option.copy()
                for _set in testsets:
                    for k in infr_option:
                        if isinstance(infr_option[k], str) and '{}' in infr_option[k]:
                            running_option[k] = infr_option[k].format(_set)
                            sys.stdout.write(fmt.format(
                                f"{_set}: {k} -> {running_option[k]}"))

                    interface.main(get_args(
                        running_option,
                        interface._parser(),
                        [running_option['nbestlist'], running_option['output']]
                    ))
            else:
                interface.main(get_args(
                    infr_option,
                    interface._parser(),
                    [infr_option['nbestlist'], infr_option['output']]
                ))
        else:
            sys.stderr.write(
                f"warning: interface '{intfname}' only support handcrafted execution.\n")

        if 'er' in infer_setting:
            backup = readfromjson(f_hyper)
            new_cfg = backup.copy()
            new_cfg['inference'] = new_cfg['inference'].copy()
            if 'avgmodel' in new_cfg['inference']:
                del new_cfg['inference']['avgmodel']
            if 'infer' in new_cfg['inference']:
                del new_cfg['inference']['infer']

            try:
                dumpjson(new_cfg, f_hyper)
                os.system(" ".join([
                    sys.executable,     # python interpreter
                    os.path.join(
                        os.path.dirname(
                            sys.argv[0]
                        ),
                        'asr.py'
                    ),        # file script
                    working_dir,
                    "--silent",
                    "--start_stage=4",
                    "--stop_stage=4",
                    f"--ngpu={args.ngpu}"
                ]))

            except Exception as e:
                raise RuntimeError(str(e))
            finally:
                dumpjson(backup, f_hyper)
