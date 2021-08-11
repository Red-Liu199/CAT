import argparse
import sentencepiece as spm
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="SPM model location.")

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.model)

    for line in sys.stdin:
        token = sp.decode([int(x) for x in line.split(' ')])
        sys.stdout.write(token+'\n')
