# build phone vocab from lexicon

import os
import sys

if __name__ == "__main__":
    try:
        lexicon = sys.argv[1]
        assert os.path.isfile(lexicon)
    except Exception:
        print(f"Usage: python {sys.argv[0]} lexicon.txt")
        sys.exit(1)

    vocab = set()
    with open(lexicon, 'r') as fit:
        for line in fit:
            content = line.strip().split()
            for phn in content[1:]:
                vocab.add(phn)

    # keep 0 for <s>/<blk>, 1 for <unk>
    vocab = {phn: str(idx+2) for idx, phn in enumerate(sorted(vocab))}
    try:
        sys.stdout.write("<s>\t0\n")
        sys.stdout.write("<unk>\t1\n")
        with open(lexicon, 'r') as fit:
            for line in fit:
                content = line.strip().split()
                sys.stdout.write(
                    "{}\t{}\n".format(
                        content[0],
                        ' '.join(vocab[p] for p in content[1:])
                    )
                )
    except IOError:
        pass
    finally:
        sys.stdout.flush()
