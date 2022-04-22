"""Parse arguments with python style
use 'null' to indicat python None in JSON

NOTE:
    1. You should acknownledge the usage of argparse module of python first.
    2. Shell var '$' symbol can be used in 'default', if the order of arguments are right. e.g.
        in the following example: 
        in the default of --output, '$input' will be replaced by <input>;
        but note that you can't refer to '$output' in input parser, where
        the '$output' is not initialized yet.
    3. The line 
        'opts=$(python utils/parseopt.py $0 $*) && eval $opts || exit 1' in
        shell script cannot be replace to 
        'eval $(python utils/parseopt.py $0 $*) || exit 1'
        the latter would produce some unexpected prompt with 'example.sh -h'
    4. To flag the start of parser, following statements are all allowed:
        4.1 <<"PARSER"  4.2 <<'PARSER'  4.3 <<PARSER  4.4 << "PARSER" ...

Usage: in shell script
example.sh:
<<"PARSER"
("input", type=str, help="Input file.")
("-o", "--output", type=str, default="${input}_out",
    help="Output file. Default: <input>_out")
PARSER
opts=$(python utils/parseopt.py $0 $*) && eval $opts || exit 1
"""

import re
import sys
import argparse

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(
            "This script is used to parse options for shell script.\n"
            f"... read header of {sys.argv[0]} for the usage.\n")
        sys.exit(1)
    script = sys.argv[1]
    argsin = sys.argv[2:]

    # match lines in '<<PARSER' in 'PARSER'
    parser_pattern = re.compile(
        r"^<<\s*(?:\"PARSER\"|'PARSER'|PARSER)\s*$((?:.|\n)*?)^\s*PARSER\s*$",
        re.MULTILINE)
    # split lines via brackets
    argument_pattern = re.compile(r"^[(]((?:.|\n)*?)[)]$", re.MULTILINE)
    with open(script, 'r') as fi:
        s = fi.read()
    parserinfo = parser_pattern.findall(s)
    match = argument_pattern.findall(parserinfo[0])

    parser = argparse.ArgumentParser(prog=script)
    for arg in match:
        # NOTE: 
        # This is potential dangerous. It's your duty to ensure the safety.
        eval(f"parser.add_argument({arg})")

    try:
        for arg, value in vars(parser.parse_args(argsin)).items():
            if isinstance(value, list):
                # deal with nargs='+' and nargs='*'
                value = f"\"{' '.join([str(x) for x in value])}\""
            sys.stdout.write(f"export {arg}={value}; ")
    except SystemExit as se:
        # re-locate the help information to error
        if se.code == 0:
            parser.print_help(sys.stderr)
        sys.exit(1)
    else:
        sys.exit(0)
