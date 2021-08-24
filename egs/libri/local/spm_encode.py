import argparse
import sentencepiece as spm
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="SPM model location.")
    parser.add_argument("--lower", action="store_true", default=False,
                        help="Lower case the string beforce encoding. Default False.")

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.model)

    for line in sys.stdin:
        if args.lower:
            line = line.lower()
        encoded_line = sp.encode(line)
        sys.stdout.write(' '.join((str(x) for x in encoded_line))+'\n')
