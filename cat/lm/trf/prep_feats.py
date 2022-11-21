"""
Author: Zheng Huahuan

Prepare pi (length information) and discrete feats
"""

from ...shared.data import (
    CorpusDataset,
    sortedPadCollateLM
)
from . import feat


import argparse
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def main(trset: str, f_linfo: str,  feat_type_file: str = None, f_feats: str = None, batch_size: int = 128):
    train_set = CorpusDataset(trset)

    if feat_type_file is not None:
        assert f_feats is not None

        trainloader = DataLoader(
            train_set,
            batch_size=batch_size, shuffle=False,
            num_workers=1,
            collate_fn=sortedPadCollateLM(False)
        )
        wftype, cftype = feat.separate_type(
            feat.read_feattype_file(feat_type_file))
        wfeat = feat.Feats(wftype)
        for i, minibatch in tqdm(enumerate(trainloader), desc=f'Counting lengths and keys',
                                 unit='batch', total=len(trainloader), leave=False):
            features, in_lens, _, _ = minibatch
            wfeat.load_from_seqs(features, in_lens)
            wfeat.create_values_buf(wfeat.num)

        with open(f_feats, 'w') as f:
            wfeat.save(f)

    Ls = train_set.get_seq_len()
    # predifine a max length
    max_len = max(Ls)+1
    max_len = max(200, max_len)
    linfo = np.zeros(max_len, dtype=np.float32)

    for l in Ls:
        if l==0:
            linfo[1]+=1
        else:
            linfo[l] += 1

    # Laplace smoothing
    linfo[1:] += 1
    linfo /= (linfo.sum()+max_len)
    # linfo[2:] += 1
    # linfo /= linfo[2:].sum(axis=0)
    with open(f_linfo, 'wb') as fob:
        pickle.dump({
            'pi': linfo,
            'max_len': max_len, 
            "mean": np.mean(Ls),
            "std": np.std(Ls)
        }, fob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trset", type=str, help="Corpus dataset.")
    parser.add_argument("f_linfo", type=str,
                        help="Output file of length info.")
    parser.add_argument("--feat_type_file", type=str)
    parser.add_argument("--f_feats", type=str,
                        help="Ouput file of discrete feats.")
    args = parser.parse_args()

    main(**vars(args))
