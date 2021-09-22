'''Parse arguments with python style
use 'null' to indicat python None in JSON
Usage: in shell script
example.sh:

opts=$(python ctc-crf/parseopt.py '{
        "--data":{
            "type": "str",
            "default": 'null',
            "help": "Directory of data"
        },
        "--weight":{
            "type": "int",
            "default": 1.0,
            "help": "Weight factor of model"
        }
    }' $0 $*) && eval $opts || exit 1
'''
import os
import sys
import json
import argparse

if __name__ == "__main__":

    arguments = sys.argv[1]
    shell = sys.argv[2]
    argsin = sys.argv[3:]

    arguments = json.loads(arguments)  # type:dict

    parser = argparse.ArgumentParser(prog=shell)
    for k, v in arguments.items():
        if 'type' in v:
            v['type'] = eval(v['type'])
        parser.add_argument(k, **v)
    del arguments, shell

    try:
        for arg, value in vars(parser.parse_args(argsin)).items():
            print(f"{arg}={value};")

    except SystemExit as se:
        # re-locate the help information to error
        if se.code == 0:
            print(parser.print_help(sys.stderr))
        sys.exit(1)

    sys.exit(0)
