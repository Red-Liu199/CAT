import pickle

if __name__ == "__main__":
    src_path = '/mnt/workspace'
    dst_path = '/home'

    for file in ['tr', 'cv']:

        with open('data/pickle/{}.pickle'.format(file), 'rb') as fi:
            dataset = pickle.load(fi)

        for i in range(len(dataset)):
            dataset[i][1] = dataset[i][1].replace(src_path, dst_path)

        with open('data/pickle/{}.pickle'.format(file), 'wb') as fo:
            pickle.dump(dataset, fo)
