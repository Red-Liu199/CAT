"""
Resolve the data location from CAT.

e.g.

path_to_transducer/egs/wsj$ python utils/data/resolvedata.py

'wsj' will be recognized as the name of the recipe
if data/src/ does not exist:
    try to fine ../../tools/CAT/egs/wsj
    link ../../tools/CAT/egs/wsj/data to data/src

find datasets in data/src, which should satisfy:
    1. data/src/SET and data/src/SET/text exist
    2. data/src/SET/feats.scp exist

the found datasets info would be stored at data/metainfo.json in JSON format
    {
        "set-0":{
            "scp": "path/to/scp-0",
            "trans" : "path/to/trans-0"
        },
        ...
    }

all following pipeline would depend on data/metainfo.json, you can modify it manually for flexible usage
"""

import os
import sys
import glob
import json
from typing import Dict, List

D_SRCDATA = 'data/src'


def find_dataset(d_data: str) -> Dict[str, Dict[str, str]]:
    assert os.path.isdir(d_data), f"'{d_data}' is not a directory."

    datasets = {}
    for d in os.listdir(d_data):
        _setdir = os.path.join(d_data, d)
        if not os.path.isdir(_setdir):
            continue

        check_cnt = 0
        for f in os.listdir(_setdir):
            if f == 'feats.scp':
                check_cnt += 1
            elif f == 'text':
                check_cnt += 1

            if check_cnt == 2:
                break
        if check_cnt != 2:
            continue

        datasets[d] = {
            'scp': os.path.abspath(os.path.join(_setdir, 'feats.scp')),
            'trans': os.path.abspath(os.path.join(_setdir, 'text')),
        }
    return datasets


def main():
    from cat.shared._constants import F_DATAINFO
    found_datasets = {}
    if os.path.isdir(D_SRCDATA):
        found_datasets.update(find_dataset(D_SRCDATA))
    else:
        recipe = os.path.basename(os.getcwd())
        src_recipedata = f"../../tools/CAT/egs/{recipe}/data"
        if not os.path.isdir("../../tools/CAT"):
            print("warning: tool/CAT is not linked to ../../tools/CAT")
        elif not os.path.isdir(src_recipedata):
            print(f"'{recipe}' is not found under ../../tools/CAT/egs")
        else:
            os.makedirs(os.path.dirname(D_SRCDATA), exist_ok=True)
            os.system(f"ln -s {os.path.abspath(src_recipedata)} {D_SRCDATA}")
            found_datasets.update(find_dataset(src_recipedata))

    if os.path.isfile(F_DATAINFO):
        backup = json.load(open(F_DATAINFO, 'r'))
        meta = backup.copy()
    else:
        os.makedirs(os.path.dirname(F_DATAINFO), exist_ok=True)
        backup = None
        meta = {}

    meta.update(found_datasets)
    try:
        with open(F_DATAINFO, 'w') as fo:
            json.dump(meta, fo, indent=4, sort_keys=True)
    except Exception as e:
        if backup is not None:
            with open(F_DATAINFO, 'w') as fo:
                json.dump(backup, fo, indent=4, sort_keys=True)
        raise RuntimeError(str(e))


if __name__ == "__main__":
    main()
