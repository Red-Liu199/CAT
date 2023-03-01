import pickle
import os
import argparse
import torch
import copy

def delete_ids(input_dir, output_dir, head_del_num, tail_del_num):
    for set in ['train', 'dev', 'test']:
        _seeks = []
        count=0
        if not os.path.exists(os.path.join(input_dir,'{}.pkl.bin'.format(set))):
            continue
        with open(os.path.join(output_dir,'{}.pkl.bin'.format(set)), 'wb') as fo:
            with open(os.path.join(input_dir, '{}.pkl.bin'.format(set)), 'rb') as fi:
                while(fi.tell()<os.fstat(fi.fileno()).st_size):
                    data=pickle.load(fi)
                    if isinstance(data, tuple):
                        if tail_del_num==0:
                            new_data=(data[0][head_del_num:], data[1][head_del_num:])
                        else:
                            new_data=(data[0][head_del_num:-tail_del_num], data[1][head_del_num:-tail_del_num])
                    elif isinstance(data, list):
                        new_data=data[head_del_num:-tail_del_num] if tail_del_num!=0 else data[head_del_num:]
                    _seeks.append(fo.tell())
                    pickle.dump(new_data, fo)
                    count+=1
        print('{} size {}'.format(set, count))
        with open(os.path.join(output_dir, '{}.pkl'.format(set)), 'wb') as fo:
            # save the file name of binary file
            pickle.dump(os.path.basename('{}.pkl.bin'.format(set)), fo)
            # save the location information
            pickle.dump(_seeks, fo)
            # dump a nonetype to ensure previous operation is done
            pickle.dump(None, fo)

def truncate_seqs(input_dir, output_dir, threshold, truncated_len):
    for set in ['train', 'dev', 'test']:
        _seeks = []
        count=0
        if not os.path.exists(os.path.join(input_dir,'{}.pkl.bin'.format(set))):
            continue
        with open(os.path.join(output_dir,'{}.pkl.bin'.format(set)), 'wb') as fo:
            with open(os.path.join(input_dir, '{}.pkl.bin'.format(set)), 'rb') as fi:
                while(fi.tell()<os.fstat(fi.fileno()).st_size):
                    data=pickle.load(fi)
                    if isinstance(data, tuple):
                        while(len(data[0])>threshold):
                            new_data=(data[0][:truncated_len], data[1][:truncated_len])
                            _seeks.append(fo.tell())
                            pickle.dump(new_data, fo)
                            count+=1
                            data=(data[0][truncated_len:], data[1][truncated_len:])
                    elif isinstance(data, list):
                        while(len(data)+1>threshold):
                            new_data=data[:truncated_len+1]
                            _seeks.append(fo.tell())
                            pickle.dump(new_data, fo)
                            count+=1
                            data=data[truncated_len+1:]
                    _seeks.append(fo.tell())
                    pickle.dump(data, fo)
                    count+=1
        print('{} size {}'.format(set, count))
        with open(os.path.join(output_dir, '{}.pkl'.format(set)), 'wb') as fo:
            # save the file name of binary file
            pickle.dump(os.path.basename('{}.pkl.bin'.format(set)), fo)
            # save the location information
            pickle.dump(_seeks, fo)
            # dump a nonetype to ensure previous operation is done
            pickle.dump(None, fo)

def masked_token(input_ids):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    label = torch.tensor(input_ids)
    # We sample a few tokens in each sequence for MLM training (with probability 15%)
    probabilities = torch.tensor([0.15]*len(input_ids))
    probabilities[0] = 0
    probabilities[-1] = 0
    masked_indices = torch.bernoulli(probabilities).bool()
    label[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK], id: 103)
    indices_replaced = torch.bernoulli(torch.full(label.shape, 0.8)).bool() & masked_indices
    input_ids = torch.tensor(input_ids)
    input_ids[indices_replaced] = 103

    # 10% of the time, we replace masked input tokens with random word
    current_prob = 0.1 / (1 - 0.8)
    indices_random = torch.bernoulli(torch.full(label.shape, current_prob)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(21128, label.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids.tolist(), label.tolist() 

def prepare_mlm_data(input_dir, output_dir):
    for set in ['train', 'dev', 'test']:
        _seeks = []
        count=0
        with open(os.path.join(output_dir,'{}.pkl.bin'.format(set)), 'wb') as fo:
            with open(os.path.join(input_dir, '{}.pkl.bin'.format(set)), 'rb') as fi:
                while(fi.tell()<os.fstat(fi.fileno()).st_size):
                    data=pickle.load(fi)
                    input_ids, label = masked_token(data[0])
                    new_data = (input_ids, label)
                    _seeks.append(fo.tell())
                    pickle.dump(new_data, fo)
                    count+=1
        print('{} size {}'.format(set, count))
        with open(os.path.join(output_dir, '{}.pkl'.format(set)), 'wb') as fo:
            # save the file name of binary file
            pickle.dump(os.path.basename('{}.pkl.bin'.format(set)), fo)
            # save the location information
            pickle.dump(_seeks, fo)
            # dump a nonetype to ensure previous operation is done
            pickle.dump(None, fo)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str, help="Input directory")
    parser.add_argument("outdir", type=str, help="Ouput directory")
    parser.add_argument("--head_del", type=int, default=0, help="Numbers of tokens to be deleted in the head")
    parser.add_argument("--tail_del", type=int, default=0, help="Numbers of tokens to be deleted in the tail")
    parser.add_argument("--truncate", action="store_true", help="truncate sequences")
    parser.add_argument("--truncate_threshold", type=int, default=32, help="Length threshold at which the sequence needs to be truncated")
    parser.add_argument("--truncate_length", type=int, default=16, help="Sequence length after being truncated")
    parser.add_argument('--mlm', action="store_true", help="prepare for training masked language model")
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if args.mlm:
        prepare_mlm_data(args.indir, args.outdir)
    elif args.truncate:
        truncate_seqs(args.indir, args.outdir, args.truncate_threshold, args.truncate_length)
    else:
        delete_ids(args.indir, args.outdir, args.head_del, args.tail_del)
    