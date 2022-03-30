"""
Resolve the data location from CAT.

e.g.

path_to_transducer/egs/wsj$ python utils/resolvedata.py

'wsj' will be recognized as the name of the recipe
if data/src/ does not exist:
    try to fine ../../tools/CAT/egs/wsj
    link ../../tools/CAT/egs/wsj/data to data/src

find datasets in data/src, which should satisfy:
    1. data/src/SET and data/src/SET/text exist
    2. data/src/all_ark/SET.scp exist

the found datasets info would be stored at data/.CATDATA.info in JSON format
    {
        "set-0":{
            "scp": "path/to/scp-0",
            "trans" : "path/to/trans-0"
        },
        ...
    }

all following pipeline would depend on data/.CATDATA.info, you can modify it manually for flexible usage
"""

import os
import sys
import glob
import json
from typing import Dict, List

F_DATAINFO = 'data/.CATDATA.info'
D_SRCDATA = 'data/src'


def find_dataset(d_data: str) -> Dict[str, Dict[str, str]]:
    assert os.path.isdir(d_data), f"'{d_data}' is not a directory."

    f_scps = glob.glob(f"{d_data}/all_ark/*.scp")
    if len(f_scps) == 0:
        return {}

    datasets = {}
    for file in f_scps:
        _setname = os.path.basename(file).removesuffix('.scp')
        _setdir = os.path.join(d_data, _setname)
        _settrans = os.path.join(_setdir, 'text')
        if os.path.isdir(_setdir) and os.path.isfile(_settrans):
            datasets[_setname] = {
                'scp': os.path.abspath(file),
                'trans': os.path.abspath(_settrans)
            }
    return datasets


def main():
    found_datasets = {}
    if os.path.isdir(D_SRCDATA):
        found_datasets.update(find_dataset(D_SRCDATA))
    else:
        recipe = os.path.basename(os.getcwd())
        src_recipedata = f"../../tools/CAT/egs/{recipe}/data"
        if not os.path.isdir("../../tools/CAT"):
            print("tool:CAT is not linked to ../../tools/CAT")
        elif not os.path.isdir(src_recipedata):
            print(f"'{recipe}' is not found under ../../tools/CAT/egs")
        else:
            os.system(f"ln -s {os.path.abspath(src_recipedata)} {D_SRCDATA}")
            found_datasets.update(find_dataset(src_recipedata))

    os.makedirs(os.path.dirname(F_DATAINFO), exist_ok=True)
    with open(F_DATAINFO, 'w') as fo:
        json.dump(found_datasets, fo, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
