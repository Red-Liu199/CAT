import pickle
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="inpkl", nargs='+', type=str)
    parser.add_argument("-o", dest="outpkl", type=str)

    args = parser.parse_args()

    assert isinstance(args.inpkl, list)
    assert len(args.inpkl) > 0

    datasets = []
    for pkl in args.inpkl:
        assert os.path.isfile(pkl)
        print("> Merging {}".format(pkl))
        with open(pkl, 'rb') as fi:
            datasets += pickle.load(fi)

    print("> Export to {}".format(args.outpkl))
    with open(args.outpkl, 'wb') as fo:
        pickle.dump(datasets, fo)
