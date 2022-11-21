# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Haochen Liu, Huahuan Zheng (maxwellzh@outlook.com)

from os import truncate
from ...shared.decoder import *
from ...shared.encoder import *
from ...shared import tokenizer as tknz
from . import feat

import pickle
import math
import numpy as np
from typing import *


import torch
import torch.nn as nn
import torch.nn.functional as F
from cat.shared import coreutils
from scipy.stats import norm
from cat.lm import lm_builder


class TRFLM(AbsDecoder):
    def __init__(
        self,
        noise_rate: float = 1.,  # rate of noise data number/ real data number
        # set it to "nce" or "dnce". suggest to use dnce, more completed in ploting
        method: Literal['nce', 'dnce'] = "nce",
        feat_disc: bool = False,  # discrete feature
        feat_nn: bool = True,  # neural network feature
        energy_func: str = 'sumtargetlogit', # 'hidden2scalar'/'logsumexplogit'/'maxlogit'/'sumtargetlogit'
        episilon: float = 1e-30,  # min of log()
        # length info file, refer to trf/prep_feats.py
        f_linfo: str = None,
        # postion of the document saving the discrete grammatical features
        feat_type_file: str = None,
        f_feats: str = None,
        alpha: float = 0.25,  # dnce alpha,
        config_noise_model: str = None, # noise configuration file path
        config_trf_model: str =None, # TRF model configuration file path
        check_trf_model: str =None, # load energy model from this checkpoint if its not None
        check_noise_model: str = None, # load noise model from this checkpoint if its not None
        noise_score: bool = False, # use noise model to score sentences during evaluation
        with_end_mark: bool = True,
        tokenizer_path: str = None,
        Zeta_factor: float =None, # Initialization factor for Zeta
        bert_tokenizer: bool = False # the data is encoded by bert tokenizer, with [CLS] in the head and [SEP] in the tail
        ):
        super().__init__()

        # assign settings for TRF nce training
        self.noise_rate = noise_rate
        self.method = method
        self.feat_disc = feat_disc
        self.feat_nn = feat_nn
        self.energy_func = energy_func
        self.episilon = episilon
        self.alpha = alpha
        self.noise_score = noise_score
        self.with_end_mark = with_end_mark
        self.tokenizer = tknz.load(tokenizer_path) if tokenizer_path else None
        self.bert_tokenizer = bert_tokenizer

        # initialize trf and noise model
        noise_config = coreutils.readjson(config_noise_model)
        self.noise_type = noise_config['decoder']['type']
        trf_config = coreutils.readjson(config_trf_model)
        if self.feat_nn:
            self.nn_type = list(trf_config.keys())[0] # encoder or decoder
            if check_trf_model is not None:
                # FIXEME (liuhong): 
                # This does not work when loading a encoder trf model
                trf_model = lm_builder(trf_config, dist=False)
                coreutils.load_checkpoint(trf_model, check_trf_model)
                self.udlying_nn = trf_model.lm
            else:
                model_cls = eval(trf_config[self.nn_type]['type'])
                self.udlying_nn = model_cls(**trf_config[self.nn_type]['kwargs'])

        if check_noise_model is not None:
            nlm = lm_builder(noise_config, dist=False)
            coreutils.load_checkpoint(nlm, check_noise_model)
            self.noise_model = nlm.lm
        else:
            model_cls = eval(noise_config['decoder']['type'])
            self.noise_model = model_cls(**noise_config['decoder']['kwargs'])
        
        # initialize Pi and Zeta for TRF model
        with open(f_linfo, 'rb') as fib:
            # load length information for trans-dimensional random field (TRF)
            linfo = pickle.load(fib)
        self.max_len = linfo['max_len']
        if Zeta_factor is None:
            # no zeta factor specified, use log(vocab_size) as zeta factor
            num_classes = self.udlying_nn.config.vocab_size if trf_config[self.nn_type]['type']=='PretrainedTransformer' \
                else trf_config[self.nn_type]['kwargs']['num_classes']
            self.zeta = nn.Parameter(np.log(num_classes)*torch.tensor(range(-1, self.max_len-1)))
        else:
            self.zeta = nn.Parameter(Zeta_factor*torch.tensor(range(-1, self.max_len-1)))
        self.zeta[0].data.zero_()
        self.pi = nn.Parameter(torch.tensor(linfo['pi'], dtype=torch.float32))
        # len_distribution = norm(linfo['mean'], linfo['std'])
        # self.pi = nn.Parameter(torch.tensor([len_distribution.pdf(i) for i in range(self.max_len)]))
        self.pi_noise_model = nn.Parameter(torch.tensor(linfo['pi'], dtype=torch.float32))
        # self.pi_noise_model = nn.Parameter(torch.tensor([len_distribution.pdf(i) for i in range(self.max_len)]))
        self.pi.requires_grad_(False)
        self.pi_noise_model.requires_grad_(False)
        if self.method == 'nce':
            # freeze noise model if nce training
            self.noise_model.requires_grad_(False)
            self.noise_module = [self.noise_model]
            self.noise_model = None
        else:
            self.noise_module = [self.noise_model]
        if self.energy_func!='sumtargetlogit':
            # the trf model must be an encoder or pretrained bert if energy function is not sum-target-logit
            assert self.nn_type=='encoder' or 'Bert' in trf_config[self.nn_type]['kwargs'].get('model_name', ''),\
                'Currently the TRF model must be Bert for other energy funciton'
        else:
            assert self.nn_type=='decoder', 'The TRF model must be a decoder if using the sum-target-logit energy function'
        if 'hidden2scalar' in self.energy_func:
            hidden_size = self.udlying_nn.config.hidden_size if hasattr(self.udlying_nn, 'config') else self.udlying_nn.dim_hid
            self.energy_lin = nn.Linear(in_features=hidden_size, out_features=1)
            if self.energy_func=='hidden2scalar-sum' and hasattr(self.udlying_nn, 'model') and hasattr(self.udlying_nn.model, 'pooler'):
                self.udlying_nn.model.pooler = None

        # load descrete feature
        if self.feat_disc:
            wftype, cftype = feat.separate_type(
                feat.read_feattype_file(feat_type_file))
            self.wfeat = feat.Feats(wftype)
            if f_feats is not None:
                with open(f_feats, 'r') as fi:
                    self.wfeat.restore(fi)

    # get NN feature
    def get_nn_feat(self, input_ids, logits, targets, in_lens: torch.LongTensor):
        # targets: (N, L, K)
        # logits: the output of nnlm, (N, L, V)
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)        
        w = logits.gather(index=targets, dim=-1)
        # find the length and mask the tail
        padding_mask = torch.arange(input_ids.size(1), device=input_ids.device)[
            None, :] < in_lens[:, None].to(input_ids.device)
        padding_mask = padding_mask.unsqueeze(2)
        w *= padding_mask
        # w: NN feature of the N sentences in this batch
        # w: (N, L, K)
        return w

    def _get_log_prob_w_phi(self, inputs: torch.Tensor, in_lens: torch.Tensor, targets: torch.Tensor):
        # get the log prob of N sentences
        energy = self.calculate_energy(inputs, targets, in_lens) # (B, )
        phi = -energy-self.zeta[in_lens] # (B, )
        out = phi + torch.log(self.pi[in_lens]) # (B, )
        return out, phi

    def score(self, input_ids: torch.LongTensor, targets: torch.LongTensor, in_lens: torch.Tensor, *args):
        if self.bert_tokenizer and input_ids[0][0]==0: 
            # the input sequence need to be processed:
            # input: 0[CLS]abcde[SEP] --> [CLS]abcde[SEP]
            # target: [CLS]abcde[SEP]0 --> abcde[SEP]0
            input_ids = input_ids[:, 1:] # delete 0 in the head
            targets = targets[:, 1:]
            in_lens -= 1
        if self.noise_score:
            score = self.noisem_score(input_ids, in_lens, targets)
        else:
            score, _ = self._get_log_prob_w_phi(input_ids, in_lens, targets)
        return score

    def gettrfdata(self, num: int = 10, turn_num: int = 100):
        """
        The text used to generate TRF using MIS. 
        num is the number of channels used during generation 
        (number of generated sentences), 
        turn_num is the number of cycles run.
        """
        lendata = torch.multinomial(self.pi, num, True) 
        n = 0
        # FIXME (huahuan): why here quantize the lendata to integer?
        maxlendata = int(max(lendata))
        # sentence slected in trf_data
        trfdata = torch.zeros([num, maxlendata], device=next(
            self.noise_module[0].parameters()).device, dtype=torch.long)
        # log prob of the slected sentence
        trfdata_log_pm = torch.zeros([num], device=next(
            self.noise_module[0].parameters()).device, dtype=torch.long)
        # log prob of the noise sentence
        trfdata_log_pn = torch.zeros([num], device=next(
            self.noise_module[0].parameters()).device, dtype=torch.long)
        for time in range(turn_num): # MIS iteration
            noise = torch.zeros([num, maxlendata], device=next(
                self.noise_module[0].parameters()).device, dtype=torch.long)
            ones = torch.ones([num], device=next(
                self.noise_module[0].parameters()).device, dtype=torch.long)
            noise_next = torch.zeros([num, 1], device=next(
                self.noise_module[0].parameters()).device, dtype=torch.long)
            cache = None
            for i in range(maxlendata-1):
                with torch.no_grad():
                    noise_out, cache = self.noise_module[0](
                        noise_next, cache=cache, input_lengths=ones)
                noise_out = noise_out[:, 0, :]
                noise_distribution = F.softmax(noise_out, dim=-1)
                noise_next = torch.multinomial(noise_distribution, 1, True) # sampling by probablity
                noise[:, i+1] = noise_next.squeeze(1)
            padding_mask = torch.arange(noise.size(1), device=noise.device)[
                None, :] < lendata[:, None].to(noise.device)
            noise *= padding_mask
            tar = torch.cat(
                [noise[:, 1:maxlendata], noise[:, 0].unsqueeze(1)], dim=1).unsqueeze(2)
            log_pm, phi = self._get_log_prob_w_phi(noise, lendata, tar)
            log_pn = self.noisem_score(noise, lendata, tar)
            if time == 0:
                trfdata_log_pm = log_pm
                trfdata_log_pn = log_pn
                trfdata = noise
            else:
                p = -trfdata_log_pm+log_pm+trfdata_log_pn-log_pn
                p = p.exp()
                rand = torch.rand([num], device=next(
                    self.noise_module[0].parameters()).device)
                for j in range(num):
                    # compute the rata of new sentence and old sentence, as the prob of replacing
                    if rand[j] < p[j]:
                        trfdata_log_pm[j] = log_pm[j]
                        trfdata_log_pn[j] = log_pn[j]
                        trfdata[j, :] = noise[j, :]
                        n += 1

    def getnoise(self, noise_num: int):
        # generate noise sentence，noise_num is the number of noise
        # generate the required sentence length with a priori probability
        lennoise = torch.multinomial(self.pi_noise_model, noise_num, True).to(
            next(self.noise_module[0].parameters()).device)
        maxlennoise = int(max(lennoise))
        noise = torch.zeros([noise_num, maxlennoise], device=next(self.noise_module[0].parameters()).device, dtype=torch.long)
        # When a token is generated in each round, the input length of each sentence is 1 (the rest are saved in the cache), and ones is the new input length
        ones = torch.ones([noise_num], device=next(
            self.noise_module[0].parameters()).device, dtype=torch.long)
        # predict next noise token
        if self.bert_tokenizer:
            # initialize the start token id with [CLS] id (101)
            noise_next = 101*torch.ones([noise_num, 1], device=next(
                self.noise_module[0].parameters()).device, dtype=torch.long)
            noise[:, 0] = 101*torch.ones([noise_num], device=next(
                self.noise_module[0].parameters()).device, dtype=torch.long)
        else:
            noise_next = torch.zeros([noise_num, 1], device=next(
                self.noise_module[0].parameters()).device, dtype=torch.long)
        extra_tokens = 2 if self.bert_tokenizer else 1
        generation_times = maxlennoise-extra_tokens
        noise_probs = torch.zeros([noise_num, generation_times], device=next(self.noise_module[0].parameters()).device, dtype=torch.float32)
        cache = None
        for i in range(generation_times):
            with torch.no_grad():
                if self.noise_type=='PretrainedTransformer':
                    noise_out, cache = self.noise_module[0](noise_next, cache=cache, use_cache=True)
                else:
                    noise_out, cache = self.noise_module[0](src_ids=noise_next, cache=cache, input_lengths=ones)
            noise_out = noise_out[:, 0, :]

            noise_distribution = F.softmax(noise_out, dim=-1)
            noise_next = torch.multinomial(noise_distribution, 1, True)
            probs = noise_distribution.gather(index=noise_next, dim=-1).squeeze()
            noise_probs[:, i] = probs
            noise[:, i+1] = noise_next.squeeze(1)
    
        padding_mask = torch.arange(noise.size(1), device=noise.device)[
            None, :] < lennoise[:, None].to(noise.device)
        noise *= padding_mask
        if self.bert_tokenizer:
            end_tokens = 102*noise.new_ones(noise.shape)
            # add end tokens
            noise.scatter_(-1, (lennoise-1).unsqueeze(-1), end_tokens)
        padding_mask_for_probs = torch.arange(noise_probs.size(1), 
            device=noise.device)[None, :] < (lennoise-extra_tokens)[:, None].to(noise.device)
        noise_probs = (torch.log(noise_probs)*padding_mask_for_probs).sum(dim=-1) + torch.log(self.pi[lennoise])
        # noise sample for bert: [CLS]abcde[SEP], lennoise: 5+2, noise_probs: the probs of abcede
        # noise sample for others: 0abcde, lennoise: 5+1
        return noise, lennoise, noise_probs

    def noisem_score(self, seqs, in_lens, targets):
        if targets.dim()==3:
            targets = targets.squeeze(2)
        if self.bert_tokenizer:
            truncate_num = 1 if self.with_end_mark else 2 # delete the last token and [SEP] if no end mark
        else:
            truncate_num = 0 if self.with_end_mark else 1
        noise_in_lens = in_lens - truncate_num
        log_probs = self.noise_module[0].score(seqs, targets, noise_in_lens)
        if self.noise_module[0].pi is None:
            log_probs += torch.log(self.pi[in_lens])
        return log_probs

    def cal_loss(self, inputs: torch.Tensor, energy_values, in_lens, targets):
        """
        input and target sample:
        For Bert: [CLS]abcde[SEP]  abcde[SEP]0
        For others: 0abcde  abcde0
        """

        data_sample_num = energy_values.shape[0]
        noise_sample_num = int(data_sample_num * self.noise_rate)

        if targets.dim() == 2:
            targets = targets.unsqueeze(2)

        if self.method == "nce":
            phi = -energy_values-self.zeta[in_lens]

            log_pm = phi + torch.log(self.pi[in_lens])
    
            log_pn = self.noisem_score(inputs, in_lens, targets)
            # ppl_data = torch.exp(-log_pm.sum()/in_lens.sum())
            with torch.no_grad():
                p1 = torch.sigmoid(math.log(self.noise_rate)- log_pm + log_pn)
            loss_data = -(p1*phi).mean(dim=0)

            seqs, seqlens, log_pn = self.getnoise(noise_sample_num)
            seq_targets = seqs[:, 1:] # (B, T-1)
            log_pm, phi = self._get_log_prob_w_phi(seqs, seqlens, seq_targets)
            with torch.no_grad():
                p0 = torch.sigmoid(-math.log(self.noise_rate) + log_pm - log_pn)
            loss_noise = self.noise_rate*(p0*phi).mean(dim=0)
            acc_data = (p1.data < 0.5).sum()/data_sample_num
            acc_noise = (p0.data < 0.5).sum()/noise_sample_num
            loss = loss_data + loss_noise
            return loss, \
                {
                    'train/loss_data': loss_data.detach(),
                    'train/loss_noise': loss_noise.detach(),
                    'train/acc_data': acc_data.detach(),
                    'train/acc_noise': acc_noise.detach()
                }

        elif self.method == "dnce":
            # noise in noise data
            noise_num_real = int(noise_sample_num/self.alpha) # B2
            # number of extra noise need to use in data 
            data_noise_num = int(data_sample_num*(1-self.alpha)/self.alpha) # B1

            phi = -energy_values-self.zeta[in_lens]
            log_pm = phi + torch.log(self.pi[in_lens])
            ppl_data = torch.exp(-log_pm.sum()/in_lens.sum())
            log_pn = self.noisem_score(inputs, in_lens, targets)
            log_prob_data_sum = log_pm.sum().detach()
            log_prob_noise_sum = log_pn.sum().detach()
            # for training noise model
            # Minimize KL divergence between p_d and p_n
            loss_noisem_ml = -log_pn.sum()/in_lens.sum()
            ppl_data_onnoise = torch.exp(loss_noisem_ml)

            # noise processing in mixed distribution
            if data_noise_num>0: # alpah<1
                seqs, seqlens, log_pn_noise_data = self.getnoise(data_noise_num)
                seq_targets = seqs[:, 1:]
                log_pm_noise_data, phi_noise_data = self._get_log_prob_w_phi(
                    seqs, seqlens, seq_targets)
                # merge real data and noise into a mixed set
                log_pm = torch.cat([log_pm, log_pm_noise_data])
                log_pn = torch.cat([log_pn, log_pn_noise_data])
                phi = torch.cat([phi, phi_noise_data])

                helpalpha = torch.ones([int(data_sample_num/self.alpha)], device=inputs.device)
                # in binary classification, the probability corresponding to the mixed set is also replaced by the interpolation of model probability and noise probability
                log_pm = torch.logaddexp(torch.log(
                    self.alpha*helpalpha)+log_pm, torch.log((1-self.alpha)*helpalpha)+log_pn)

            with torch.no_grad():
                p1 = torch.sigmoid(math.log(self.noise_rate) - log_pm + log_pn)

            #loss_data=-(p1*phi).mean(dim=0)
            loss_data = -torch.matmul(p1, phi/data_sample_num)*self.alpha
            loss_data_true = -torch.mean(torch.log(1-p1+self.episilon))

            # noise negative sample
            seqs2, seqlens2, log_pn_noise = self.getnoise(noise_num_real)
            ppl_noise_onnoise = torch.exp(-log_pn_noise.sum()/seqlens2.sum())
            seq_targets2 = seqs2[:, 1:]
            log_pm_noise, phi_noise = self._get_log_prob_w_phi(seqs2, seqlens2, seq_targets2)
            if self.alpha<1:
                helpalpha_noise = torch.ones(
                    [noise_num_real], device=inputs.device)
                log_pm_noise = torch.logaddexp(torch.log(self.alpha*helpalpha_noise)+log_pm_noise, torch.log((1-self.alpha)*helpalpha_noise)+log_pn_noise)

            ppl_noise = torch.exp(-log_pm_noise.sum()/seqlens2.sum())
            with torch.no_grad():
                p0 = torch.sigmoid(log_pm_noise - log_pn_noise - math.log(self.noise_rate))
            loss_noise = torch.matmul(p0, phi_noise/data_sample_num)*self.alpha
            loss_noise_true = -self.noise_rate*torch.mean(torch.log(1-p0+self.episilon))
            # compute the prediction accuracy of all samples
            acc_data = sum(p1.data < 0.5)/int(p1.shape[0])
            acc_data_sample = sum(p1.data[:data_sample_num] < 0.5)/data_sample_num
            acc_data_noise = sum(p1.data[data_sample_num:] <0.5)/data_noise_num if data_noise_num>0 else torch.tensor(0)
            acc_noise = sum(p0.data < 0.5)/int(p0.shape[0])
            loss = loss_data + loss_noise + loss_noisem_ml
            return loss, \
                {
                    'train/loss_data': loss_data.detach(),
                    'train/loss_noise': loss_noise.detach(),
                    'train/loss_noise_kl': loss_noisem_ml.detach(),
                    'train/acc_data': acc_data.detach(),
                    # "train/acc_data_sample": acc_data_sample.detach(),
                    # "train/acc_data_noise": acc_data_noise.detach(),
                    'train/acc_noise': acc_noise.detach(),
                    'train/ppl_trfM_data': ppl_data.detach(),
                    'train/ppl_trfM_noise': ppl_noise.detach(),
                    'train/ppl_noiseM_data': ppl_data_onnoise.detach(),
                    'train/ppl_noiseM_noise': ppl_noise_onnoise.detach(),
                    'train/loss_data_true': loss_data_true.detach(),
                    'train/loss_noise_true': loss_noise_true.detach(),
                    'train/loss_true': loss_data_true.detach()+loss_noise_true.detach(),
                    'train/log_prob_trf': log_prob_data_sum,
                    'train/log_prob_noise': log_prob_noise_sum,
                    'train/zeta_5': self.zeta[5].cpu().item(),
                    'train/zeta_15': self.zeta[15].cpu().item(),
                    'train/zeta_25': self.zeta[25].cpu().item()
                }
        else:
            raise RuntimeError

    def calculate_energy(self, inputs, targets, input_lengths: torch.LongTensor):
        if self.energy_func=='sumtargetlogit':
            # only this type will calculate energy per token
            # so we can use token-level discrete feature
            if targets.dim() == 2:
                targets = targets.unsqueeze(2)
            if targets.shape[1]<inputs.shape[1]:
                truncate_num = inputs.shape[1] - targets.shape[1]
                inputs = inputs[:, :-truncate_num]
                input_lengths -= truncate_num
            features = 0
            if self.feat_disc:
                disc_feats = self.wfeat.seq_list_weight_tar(inputs, targets, input_lengths)
                features += disc_feats
            if self.feat_nn:
                nn_logits, _ = self.udlying_nn(inputs, input_lengths=input_lengths)
                nn_feats = self.get_nn_feat(inputs, nn_logits, targets, input_lengths)
                features += nn_feats
            padding_mask = (torch.arange(inputs.size(1), device=inputs.device)[
                None, :] < input_lengths[:, None].to(inputs.device)).unsqueeze(2)
            energy = -(features*padding_mask).sum(dim=1).squeeze(1)
        # Note: the nn model must be BERT for the following 3 energy functions
        elif 'hidden2scalar' in self.energy_func:
        # elif self.energy_func=='hidden2scalar':
            # TODO: add input length
            if self.energy_func=='hidden2scalar-sum':
                outputs = self.udlying_nn(inputs, input_lengths=input_lengths)
                assert 'last_hidden_state' in outputs, 'The outputs has no attribute last_hidden_state'
                hiddens = outputs.last_hidden_state # (B, T, H)
                energy = self.energy_lin(hiddens).squeeze(-1).sum(-1) # (B,)

            else: # default: use the hidden state of [CLS] to represent the sentence hidden
                outputs = self.udlying_nn(inputs, input_lengths=input_lengths)
                assert 'pooler_output' in outputs, 'The outputs has no attribute pooler_output'
                hidden = outputs.pooler_output # (B,H)
                energy = self.energy_lin(hidden).squeeze(-1) # (B,)

        elif self.energy_func=='logsumexplogit':
            outputs = self.udlying_nn(inputs, input_lengths=input_lengths)
            assert 'logits' in outputs, 'The output has no attribute logits'
            logit = outputs.logits[:, 0, :] # (B, Classes)
            energy = -torch.logsumexp(logit, dim=-1)
        elif self.energy_func=='maxlogit':
            outputs = self.udlying_nn(inputs, input_lengths=input_lengths)
            assert 'logits' in outputs, 'The output has no attribute logits'
            logit = outputs.logits[:, 0, :] # (B, Classes)
            energy = -torch.max(logit, dim=-1)
        elif self.energy_func=='summasklogit':
            # mask each token and obtain its logit on the original token
            # then sum all mask logits. (only for bert with LM head)
            # This energy function is more time-consuming than others
            energy = 0
            for t in range(1, inputs.shape[1]):
                masked_inputs = inputs.clone()
                masked_inputs[:, t] = 103*torch.ones([inputs.shape[0]], device=inputs.device, dtype=torch.long)
                outputs = self.udlying_nn(masked_inputs, input_lengths=input_lengths)
                assert 'logits' in outputs, 'The output has no attribute logits'
                logit = outputs.logits[:, t, :] # (B, V)
                energy += -logit.gather(index=inputs[:, t].unsqueeze(1), dim=-1).squeeze() #(B,)
        elif self.energy_func=='sumtokenlogit':
            outputs = self.udlying_nn(inputs, input_lengths=input_lengths)
            assert 'logits' in outputs, 'The output has no attribute logits'
            logits = outputs.logits # (B, T, V)
            energy = -logits.gather(index=inputs.unsqueeze(-1), dim=-1).squeeze().sum(-1) # (B,)
        else:
            raise RuntimeError
        return energy # shape: (B,)

    def forward(self, inputs, targets, input_lengths: torch.LongTensor):
        return self.calculate_energy(inputs, targets, input_lengths)
