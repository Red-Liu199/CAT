import json
import os
import numpy as np

import trie
from multiprocessing import Process, Manager, Queue, Value

import torch
import torch.nn as nn
import torch.nn.functional as F


def write_value(f, value, name='value'):
    if isinstance(value, np.int64) or isinstance(value, np.int32):
        value = int(value)
    if isinstance(value, np.float32) or isinstance(value, np.float64):
        value = float(value)
    f.write('{} = {}\n'.format(name, json.dumps(value)))


def read_value(f):
    s = f.__next__()
    idx = s.find('= ')
    return json.loads(s[idx+1:])


class getfeatvalue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, id):
        ctx.save_for_backward(values)
        ctx.id = id

        return values[id]

    @staticmethod
    def backward(ctx, grad_output):
        values = ctx.saved_tensors
        id = ctx.id
        grad_v = torch.zeros_like(values).float()
        grad_v[id] += 1
        return grad_v, None


class Feats(nn.Module):
    def __init__(self, type_dict):
        """
        create a collection of types of features
        Args:
            type_dict: the type dict, key=type name, value=cutoff list or integer, such as
                    {
                        'w[1:4]': [0, 0, 0, 2],
                        'w[1]-[1]w[2]':[2]
                    }
        """
        super(Feats, self).__init__()
        self.feat_list = []
        self.type_dict = type_dict
        for key, v in type_dict.items():
            self.feat_list.append(SingleFeat(key, v))  # 全部文法特征是所有单一文法特征的加和
        self.num = 0  # 选用文法特征的数量
        self.values = nn.Parameter()  # 直接对应相应文法特征的权值
        self.max_order = max([f.max_order for f in self.feat_list])
        self.create_values_buf(0)

    def create_values_buf(self, feat_num):
        self.num = feat_num
        self.values = nn.Parameter(torch.zeros(feat_num))

    def load_from_seqs(self, seqs, input_lengths):
        seqs = F.pad(seqs, [0, 1, 0, 0])
        for ftype in self.feat_list:
            for i in range(seqs.shape[0]):
                seq = seqs[i][0:int(input_lengths[i]+1)]
                self.num = ftype.exact(seq, self.num)

    def insert_single_feat(self, single_feat):
        """
        Insert a single feat into the package
        Args:
            single_feat: SingleFeat

        Returns:

        """
        self.feat_list.append(single_feat)
        self.num += single_feat.num
        print('[%s.%s] recreate value buf = %d.' %
              (__name__, self.__class__.__name__, self.num))
        self.create_values_buf(self.num)

    def save(self, f, cutoff=False, value_buf=None):
        if value_buf is None:
            value_buf = self.values

        write_value(f, self.type_dict, 'feature_type')
        write_value(f, self.num, 'feature_total_num')
        write_value(f, self.max_order, 'feature_max_order')
        for ftype in self.feat_list:
            if cutoff:
                ftype.write(f, value_buf)
            else:
                ftype.write_origin(f, value_buf)

    def restore(self, f, cutoff=False):
        type_dict = read_value(f)
        self.__init__(type_dict)

        self.num = read_value(f)
        self.max_order = read_value(f)
        self.create_values_buf(self.num)
        for ftype in self.feat_list:
            if cutoff:
                ftype.read(f, self.values)
            else:
                ftype.read_origin(f, self.values)

    @staticmethod
    def load(fp):
        type_dict = read_value(fp)
        f = Feats(type_dict)

        f.num = read_value(fp)
        f.max_order = read_value(fp)
        f.create_values_buf(f.num)
        for ftype in f.feat_list:
            ftype.read(fp, f.values)
        return f

    def ngram_find(self, ngrams):
        """ return a list containing the existing feature id """
        res = []
        for ngram in ngrams:
            a = []
            for ftype in self.feat_list:
                a += ftype.ngram_find([ngram])[0]
            res.append(a)
        return res

    def ngram_weight(self, ngrams):
        """
        find the feature observed in ngrams
        :param ngrams: 2D array, of size (batch_size, order), the order should be <= feat.max_order
        :return: list of list, containing all the feature id
        """
        # o = np.zeros(len(ngrams))
        # for ftype in self.feat_list:
        #     o += ftype.ngram_weight(ngrams, self.values)
        # return o

        ids = self.ngram_find(ngrams)
        return [torch.sum(self.values[x]) for x in ids]

    def seq_find(self, seq):
        """
        find the features observed in given sequence
        Args:
            seq: a list

        Returns:

        """
        a = []
        for ftype in self.feat_list:
            a += ftype.seq_find(seq)
        return a

    def seq_weight(self, seq):
        """
        Return the summation of weight of observed features
        Args:
            seq: a list

        Returns:

        """
        return torch.sum(self.values[self.seq_find(seq)])

    def seq_list_weight(self, seq_list):
        """
        Return the weight of a list of sequences
        Args:
            seq_list:

        Returns:

        """
        return [self.seq_weight(seq) for seq in seq_list]

    def seq_list_weight_tar(self, inputs, tar0, input_lengths):
        #get discrete feature
        seqout = torch.zeros_like(tar0).float()
        inputs = inputs.unsqueeze(2).repeat(1, 1, self.max_order)
        for i in range(self.max_order-1):
            inputs[:, i+1:inputs.shape[1], self.max_order-2 -
                   i] = inputs[:, :inputs.shape[1]-i-1, self.max_order-1]
            inputs[:, :i+1, self.max_order-2-i] = -1
        for ftype in self.feat_list:
            for k in range(tar0.shape[2]):
                key3D_list = ftype.seq_find_tar(inputs, tar0, k)
                for key_3D in key3D_list:
                    for i in range(inputs.shape[0]):
                        for j in range(input_lengths[i]):
                            key = key_3D[i, j, :]
                            id = ftype.trie_cutoff.find(key)
                            if id == None:
                                continue
                            seqout[i][j][k] += self.values[id]
        #seqout:[N,S,K]

        return seqout

    def seq_list_find(self, seq_list):
        return [self.seq_find(seq) for seq in seq_list]


class FastFeats(Feats):
    def __init__(self, type_dict, sub_process_num=100):
        super().__init__(type_dict)

        self.sub_process_num = sub_process_num
        self.task_queue = Queue(maxsize=100)
        self.res_queue = Queue(maxsize=100)
        self.sub_processes = [Process(target=self.sub_process, args=(self.task_queue, self.res_queue))
                              for _ in range(self.sub_process_num)]

    def __del__(self):
        if self.sub_processes[0].is_alive():
            self.release()

    def start(self):
        for p in self.sub_processes:
            p.start()

    def release(self):
        for _ in range(self.sub_process_num):
            self.task_queue.put((-1, []))

        for p in self.sub_processes:
            p.join()

    def sub_process(self, task_queue, res_queue):
        print('[FastFeat] sub-process %d, start' % os.getpid())
        while True:
            tsk = task_queue.get()  # tuple( id, seq )
            if tsk[0] == -1:
                break

            out = self.seq_list_weight_tar(tsk[1], tsk[2], tsk[3])
            res_queue.put((tsk[0], out))  # tuple( id, list )

        print('[FastFeat] sub-process %d, finished' % os.getpid())

    def seq_list_weight_tar_f(self, seq_list, tar0, input_lengths):
        seqout = torch.zeros_like(tar0).float()
        if not self.sub_processes[0].is_alive():
            self.start()

        # add task
        batch_size = int(np.ceil(seq_list.shape[0] / self.sub_process_num))
        tsk_num = 0
        for batch_beg in range(0, seq_list.shape[0], batch_size):
            self.task_queue.put((tsk_num, seq_list[batch_beg: batch_beg + batch_size],
                                tar0[batch_beg: batch_beg + batch_size], input_lengths[batch_beg: batch_beg + batch_size]))
            tsk_num += 1

        # collect results
        res_dict = {}
        for _ in range(tsk_num):
            i, x = self.res_queue.get()
            res_dict[i] = x

        for i in range(tsk_num):
            seqout[batch_size*i:batch_size*(i+1)] += res_dict[i]

        assert seqout.shape[0] == seq_list.shape[0]
        return seqout


class SingleFeat:
    def __init__(self, type, cutoff):
        """
        create a set of features
        :param type: a string denoting the feature type, such as "w[1:4]" or "w[1]-[1]w[1]"
        """
        self.type = type
        self.cutoff = cutoff
        if type != '':
            self.map_list = self.analyze_type(type)
            self.max_order = max([-np.min(m)+1 for m in self.map_list])
        self.trie_count = trie.trie()
        self.trie_cutoff = trie.trie()
        self.num = 0
        self.total_num = 0

    def analyze_type(self, type):
        # for example "w[1:4]", idx==3; w[1]-[1]w[1], idx==-1
        idx = type.find(':')
        if idx == -1:
            type_list = [type]  # type_list==["w[1]-[1]w[1]"]
        else:
            beg = type.rfind('[', 0, idx)  # beg==1
            end = type.find(']', idx)  # end==5
            v1 = int(type[beg+1: idx])  # v1==1
            v2 = int(type[idx+1: end])  # v2==4
            fmt = type[0:beg+1] + '{}' + type[end:]  # fmt=="w[]"
            # type_list==["w[1]","w[2]","w[3]","w[4]"]
            type_list = [fmt.format(i) for i in range(v1, v2+1)]

        map_list = []  # building map list, take type_list==["w[1]","w[2]","w[3]","w[4]"] or ["w[1]-[1]w[1]"] as example
        for t in type_list:
            a = filter(None, t.split(']'))
            n = 0
            p = []
            for s in a:
                i = int(s[2:])
                if s[0] != '-':
                    p += list(range(n, n+i))
                n += i
            p = np.array(p) - n + 1
            # map_list == [[0],[-1,0],[-2,-1,0],[-3,-2,-1,0]] or [[-2,0]]
            map_list.append(p)
        return map_list

    def exact_key(self, seq):
        """exact all the keys observed in the given sequence, used to find features"""
        key_list = []
        gram_list = []
        seq = np.array(seq)
        for m in self.map_list:
            n = -np.min(m) + 1
            if n == 1:
                # for unigram, skip the unigram for begin-token and end-token
                seq_revise = seq[1:-1]
                for i in range(0, len(seq_revise)-n+1):
                    # the second "list index" is for skip patterns
                    key = seq_revise[i:i+n][m+n-1]
                    key_list.append(key.tolist())
                    gram_list.append(n)
            else:
                for i in range(0, len(seq)-n+1):
                    key = seq[i:i+n][m+n-1]
                    key_list.append(key.tolist())
                    gram_list.append(n)
        return key_list, gram_list

    def exact(self, seq, beg_id=0):
        key_list, gram_list = self.exact_key(seq)
        for key, gram in zip(key_list, gram_list):
            if gram <= len(self.cutoff):
                cut_num = self.cutoff[gram-1]
            else:
                cut_num = 0
            count = self.trie_count.find(key)
            if count is None:
                self.trie_count.insert(key, 1)
                self.total_num += 1
                if cut_num == 0:
                    self.trie_cutoff.insert(key, beg_id)
                    beg_id += 1
                    self.num += 1
            else:
                self.trie_count.insert(key, count + 1)
                if count == cut_num:
                    self.trie_cutoff.insert(key, beg_id)
                    beg_id += 1
                    self.num += 1
        return beg_id

    def add_ngram(self, ngram_list, beg_id=0):
        """
        add a list of ngrams directly into the features
        Args:
            ngram_list: list of ngrams
            beg_id: the begin id

        Returns:
            the final id
        """
        # f = open('temp.txt', 'wt')
        for key in ngram_list:
            if not key:
                raise TypeError('[%s.%s] input an empty ngram key!' %
                                (__name__, self.__class__.__name__))
            sub = self.trie_cutoff.setdefault(key, beg_id)
            # f.write('{}  {}\n'.format(str(key), sub.data))
            if sub.data == beg_id:  # add successfully
                beg_id += 1
                self.total_num += 1
                self.num += 1
        # f.close()
        return beg_id

    def ngram_find(self, ngrams):
        """
        find the feature observed in ngrams
        :param ngrams: 2D array, of size (batch_size, order), the order should be <= feat.max_order
        :return: list of list, containing all the feature id
        """
        res = []
        for ngram in ngrams:
            n = len(ngram)
            ids = []
            for m in self.map_list:
                key = list(ngram[m+n-1])
                id = self.trie_cutoff.find(key)
                if id is not None:
                    ids.append(id)
            res.append(ids)
        return res

    def ngram_weight(self, ngrams, values):
        ids = self.ngram_find(ngrams)
        return [np.sum(values[np.array(x, dtype='int64')]) for x in ids]

    def seq_find(self, seq):
        """input a sequence, and find the observed features"""
        ids = []
        key_list, gram_list = self.exact_key(seq)
        for key in key_list:
            id = self.trie_cutoff.find(key)
            if id is not None:
                ids.append(id)
        return ids

    def seq_find_tar(self, inputs, tar0, K):
        inputs = torch.cat([inputs, tar0[:, :, K].unsqueeze(2)], dim=2)
        len = inputs.size(2)
        inputs = inputs.cpu().numpy()
        inputs = np.array(inputs)
        key3D_list = []
        for m in self.map_list:
            key_3D = inputs[:, :, m+len-1]
            key3D_list.append(key_3D)
        return key3D_list

    def write(self, f, values=None):
        f.write('feat_type = {}\n'.format(self.type))
        f.write('feat_cutoff = {}\n'.format(self.cutoff))
        f.write('feat_num_cut = {}\n'.format(self.num))
        f.write('feat_num_uncut = {}\n'.format(self.total_num))

        write_num = 0
        for key, count in trie.TrieIter(self.trie_count):
            if count is None:
                continue

            f.write('key={} count={}\n'.format(json.dumps(key), count))
            write_num += 1

        assert write_num == self.total_num

        write_num = 0
        for key, id in trie.TrieIter(self.trie_cutoff):
            if id is None:
                continue

            if values is not None:
                v = float(values[id])
            else:
                v = 0

            f.write('key={} id={} value={}\n'.format(json.dumps(key), id, v))
            write_num += 1

        assert write_num == self.num

    def write_origin(self, f, values=None):
        f.write('feat_type = {}\n'.format(self.type))
        f.write('feat_num = {}\n'.format(self.num))

        write_num = 0
        for key, id in trie.TrieIter(self.trie_cutoff):
            if id is None:
                continue

            if values is not None:
                v = float(values[id])
            else:
                v = 0
            f.write('key={} id={} value={}\n'.format(json.dumps(key), id, v))
            write_num += 1

        assert write_num == self.num

    def read(self, f, values=None):
        self.type = f.__next__().split()[-1]
        self.cutoff = list(f.__next__().split()[-1])
        self.__init__(self.type, self.cutoff)

        self.num = int(f.__next__().split()[-1])
        self.total_num = int(f.__next__().split()[-1])

        for i in range(self.total_num):
            s = f.__next__()
            s = s.replace('key=', '|')
            s = s.replace(' count=', '|')
            a = list(filter(None, s.split('|')))
            key = json.loads(a[0])
            count = int(a[1])
            self.trie_count.insert(key, count)

        for i in range(self.num):
            s = f.__next__()
            s = s.replace('key=', '|')
            s = s.replace(' id=', '|')
            s = s.replace(' value=', '|')
            a = list(filter(None, s.split('|')))
            key = json.loads(a[0])
            id = int(a[1])
            v = float(a[2])
            self.trie_cutoff.insert(key, id)
            if values is not None:
                values[id] = v

    def read_origin(self, f, values=None):
        self.type = f.__next__().split()[-1]
        self.__init__(self.type, [])

        self.num = int(f.__next__().split()[-1])
        for i in range(self.num):
            s = f.__next__()
            s = s.replace('key=', '|')
            s = s.replace(' id=', '|')
            s = s.replace(' value=', '|')
            a = list(filter(None, s.split('|')))
            key = json.loads(a[0])
            id = int(a[1])
            v = float(a[2])
            self.trie_cutoff.insert(key, id)
            if values is not None:
                values[id] = v


def read_feattype_file(fname):
    type_dict = dict()
    with open(fname, 'rt') as f:
        for line in f:
            commemt = line.find('//')
            if commemt != -1:
                line = line[:commemt]
            a = line.split()
            if len(a) == 0:
                continue

            type = a[0]
            cutoff = [0]
            if len(a) >= 2:
                s = a[1]
                if len(s) == 1:
                    cutoff = [int(s)]
                else:
                    cutoff = []
                    for c in s:
                        cutoff.append(int(c))
            type_dict[type] = cutoff
    return type_dict


def separate_type(type_dict):
    """return the type for word and class separately"""
    wt_dict = dict()
    ct_dict = dict()
    for key, v in type_dict.items():
        contain_word = key.find('w') != -1
        contain_class = key.find('c') != -1
        if contain_word and not contain_class:
            wt_dict[key] = v
        if contain_class and not contain_word:
            ct_dict[key] = v

    return wt_dict, ct_dict
