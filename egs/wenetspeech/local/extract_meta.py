"""
Extract the meta info.
Author: Zheng Huahuan
"""

import os
import json
import argparse
import pickle
from typing import List, Dict, Any, Tuple


def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="Directory to the WenetSpeech data.")
    parser.add_argument("meta_json", type=str, 
                        help="The JSON file contains meta information.")
    parser.add_argument("-o", type=str, dest='output_dir', default='data/thaudio',
                        help="Ouput directory. default: 'data/thaudio'")
    args = parser.parse_args()
    assert os.path.isfile(args.meta_json), args.meta_json
    assert os.access(args.meta_json, os.R_OK), args.meta_json
    assert os.path.isdir(args.data), args.data
    assert os.access(args.data, os.R_OK), args.data

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(f'{args.output_dir}/.done'):
        with open(args.meta_json, 'r') as injson:
            json_data = json.load(injson)

        with open(f'{args.output_dir}/text', 'w') as utt2text, \
                open(f'{args.output_dir}/segments', 'w') as segments, \
                open(f'{args.output_dir}/utt2subsets', 'w') as utt2subsets, \
                open(f'{args.output_dir}/wav', 'w') as wavscp:
            for long_audio in json_data['audios']:
                try:
                    long_audio_path = os.path.realpath(
                        os.path.join(args.data, long_audio['path']))
                    aid = long_audio['aid']
                    segments_lists = long_audio['segments']
                    duration = long_audio['duration']
                    assert (os.path.exists(long_audio_path))
                except AssertionError:
                    print(f'''Warning: {aid} something is wrong,
                                maybe AssertionError, skipped''')
                    continue
                except Exception:
                    print(f'''Warning: {aid} something is wrong, maybe the
                                error path: {long_audio_path}, skipped''')
                    continue
                else:
                    wavscp.write(f'{aid}\t{long_audio_path}\n')
                    for segment_file in segments_lists:
                        try:
                            sid = segment_file['sid']
                            start_time = segment_file['begin_time']
                            end_time = segment_file['end_time']
                            dur = end_time - start_time
                            text = segment_file['text']
                            segment_subsets = segment_file["subsets"]
                        except Exception:
                            print(f'''Warning: {segment_file} something
                                        is wrong, skipped''')
                            continue
                        else:
                            utt2text.write(f'{sid}\t{text}\n')
                            segments.write(
                                f'{sid}\t{aid}\t{start_time}\t{end_time}\n'
                            )
                            segment_sub_names = " ".join(segment_subsets)
                            utt2subsets.write(
                                f'{sid}\t{segment_sub_names}\n')
        touch(f'{args.output_dir}/.done')

    if not os.path.exists(f"{args.output_dir}/subsetlist.pkl"):
        subset = [
            'DEV',
            'L',
            'M',
            'S',
            'TEST_MEETING',
            'TEST_NET',
            'W'
        ]
        utt_in_subset = {_s: [] for _s in subset}
        print("> getting subset info...")
        with open(os.path.join(args.output_dir, 'utt2subsets'), 'r') as utt2subset:
            for line in utt2subset:
                uid, _sets = line.strip().split(maxsplit=1)
                _sets = set(_sets.split())
                for _s, _utt in utt_in_subset.items():
                    if _s in _sets:
                        _utt.append(uid)

        with open(f"{args.output_dir}/subsetlist.pkl", 'wb') as fob:
            pickle.dump(utt_in_subset, fob)
        print("> subset info got:")
        print(
            '\n'.join(
                f"  {_s}: # of utt {len(_utt)}"
                for _s,  _utt in utt_in_subset.items())
        )
        del utt_in_subset

    if not os.path.exists(f"{args.output_dir}/audiodur.pkl"):
        print("> map audio files to durations...")
        audio2uttdur = {}   # type: Dict[str, List[Tuple[str, int, int]]]
        utt2audio = {}      # type: Dict[str, str]
        with open(os.path.join(args.output_dir, 'segments'), 'r') as fit_seg:
            for line in fit_seg:
                # e.g. Y0000000000_--5llN02F84_S00000  Y0000000000_--5llN02F84 20.08   24.4
                uid, aid, s_beg, s_end = line.strip().split()
                if aid not in audio2uttdur:
                    audio2uttdur[aid] = [
                        (uid, int(float(s_beg)*100), int(float(s_end)*100))]
                else:
                    audio2uttdur[aid].append(
                        (uid, int(float(s_beg)*100), int(float(s_end)*100))
                    )
                utt2audio[uid] = aid

        with open(f"{args.output_dir}/audiodur.pkl", 'wb') as fob:
            pickle.dump(audio2uttdur, fob)
            pickle.dump(utt2audio, fob)
        print(
            "> mapping done.\n"
            f"  {len(audio2uttdur)} audios mapped."
        )
        del audio2uttdur

    if not os.path.exists(f"{args.output_dir}/corpus.pkl"):
        print("> Prepare text corpus...")
        text_corpus = {}
        with open(f"{args.output_dir}/text", 'r') as fi:
            for line in fi:
                uid, utt = line.split(maxsplit=1)
                text_corpus[uid] = utt
        with open(f'{args.output_dir}/corpus.pkl', 'wb') as fob:
            pickle.dump(text_corpus, fob)
        print("> done.")
        del text_corpus
