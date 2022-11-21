"""
Split Chinese words into characters but keep that of Latin
Author: Zheng Huahuan
"""

import sys
import os
import re
import argparse


def iszh(c: str):
    return ('\u4e00' <= c <= '\u9fa5') or ('\u3400' <= c <= '\u4db5')


def normalize_text(s_src: str):
    content = s_src.strip().split(maxsplit=1)
    if len(content) == 1:
        uid = content[0]
        return uid, ''
    elif len(content) == 2:
        uid, utt = content
        o_str = []
        utt = list(utt)
        for i in range(len(utt)-1):
            o_str.append(utt[i])
            if iszh(utt[i]) ^ iszh(utt[i+1]):
                o_str.append(' ')
        o_str.append(utt[-1])
        return uid, ''.join(o_str)
    else:
        return None, None


def rm_space_in_zh(s: str):
    if s == '':
        return s

    o_str = [s[0]]
    s = list(s)
    for i in range(1, len(s)-1):
        if iszh(s[i-1]) and iszh(s[i+1]) and s[i] == ' ':
            continue
        o_str.append(s[i])
    o_str.append(s[-1])
    return ''.join(o_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, default='', nargs='?',
                        help="Read from file. Assume each line starts with an utterance ID.")
    parser.add_argument("--keep-space", action="store_true",
                        help="Keep space between Chinese charaters, otherwise which would be removed.")
    args = parser.parse_args()

    if args.text in ('', '/dev/stdin'):
        usestd = True
    else:
        usestd = False

    if not usestd:
        r_specifier = open(args.text, 'r')
    else:
        r_specifier = sys.stdin

    p_rm_consecutive_spaces = re.compile(r"\s{2,}")
    rms = (not args.keep_space)
    try:
        for line in r_specifier:
            # add space between chinese and english
            uid, utt = normalize_text(line)
            if uid is None or utt == '':
                continue
            # rm consecutive spaces
            utt = re.sub(p_rm_consecutive_spaces, ' ', utt)
            if rms:
                # rm space between chinese charaters
                utt = rm_space_in_zh(utt)
                # add a space if the first char is not Chinese
                if not iszh(utt[0]):
                    utt = ' ' + utt

            # use lower cased
            sys.stdout.write(f"{uid}\t{utt.lower()}\n")
    except IOError:
        pass
    finally:
        if not usestd:
            r_specifier.close()
