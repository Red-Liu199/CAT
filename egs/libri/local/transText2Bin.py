import argparse
import pickle
import os
import string
import random
from tqdm import tqdm


def randName(L: int) -> str:
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(L))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("intext", type=str,
                        help="Input text files (in id format).")
    parser.add_argument("outbin", type=str, help="Ouput file.")
    parser.add_argument("--strip", action="store_true", default=False)

    args = parser.parse_args()

    if not os.path.isfile(args.intext):
        raise FileNotFoundError(f"{args.intext} does not exist!")

    dataset = []
    binfile = args.outbin+randName(8)
    with open(args.intext, 'r') as fi:
        with open(binfile, 'wb') as fo:
            for i, line in tqdm(enumerate(fi)):
                if args.strip:
                    data = [int(x) for x in line.split()[1:]]
                else:
                    data = [int(x) for x in line.split()]

                dataset.append(fo.tell())
                pickle.dump(data, fo)

    with open(args.outbin, 'wb') as fo:
        # save the file name of binary file
        pickle.dump(binfile, fo)
        # save the location information
        pickle.dump(dataset, fo)
