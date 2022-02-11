"""Parse arguments with python style
use 'null' to indicat python None in JSON

Usage: in shell script
example.sh:
<<"PARSER"
{
    "data":{
        "type": "str",
        "help": "Directory of data"
    },
    "--weight":{
        "type": "int",
        "default": 1.0,
        "help": "Weight factor of model"
    }
}
PARSER
opts=$(python utils/parseopt.py $0 $*) && eval $opts || exit 1

"""

import re
import sys
import json
import argparse

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(
            "This script is used to parse options for shell script.\n"
            f"... read header of {sys.argv[0]} for the usage.\n")
        sys.exit(1)
    script = sys.argv[1]
    argsin = sys.argv[2:]

    parseinfo = ""
    read_flag = False
    sol = 0
    with open(script, 'r') as fi:
        for line in fi:
            if line.strip().replace(' ', '') == "PARSER":
                break
            if read_flag:
                parseinfo += line
                continue
            # in case '<<"PARSER"' '<< "PARSER"' '<<PARSER'...
            if line.strip().translate(
                    {' ': '', '\'': '', '\"': ''}) == "<<\"PARSER\"":
                read_flag = True
            sol += 1

    if parseinfo == "":
        parseinfo = "{}"
    try:
        arguments = json.loads(parseinfo)  # type:dict
    except json.decoder.JSONDecodeError as err:
        # something like
        # Expecting ',' delimiter: line 24 column 5 (char 542)
        errinfo = str(err)
        errline = re.search(r"(?<=line\s)\d+", errinfo)
        if errline is not None:
            errline = int(errline.group(0))
            errinfo = errinfo.replace(f"line {errline}", f"line {errline+sol}")
        sys.stderr.write(
            "Parsing string format error:\n\t" +
            errinfo + '\n')
        sys.exit(1)

    parser = argparse.ArgumentParser(prog=script)
    for k, v in arguments.items():
        if 'type' in v:
            v['type'] = eval(v['type'])
        parser.add_argument(k, **v)
    del arguments, script

    try:
        for arg, value in vars(parser.parse_args(argsin)).items():
            if isinstance(value, list):
                # deal with nargs='+' and nargs='*'
                value = '\"'+' '.join([str(x) for x in value])+'\"'
            print(f"{arg}={value};")

    except SystemExit as se:
        # re-locate the help information to error
        if se.code == 0:
            parser.print_help(sys.stderr)
        sys.exit(1)

    sys.exit(0)
