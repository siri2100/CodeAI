import csv

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize


class Data_V10(nn.Module):
    def __init__(self, split='train', maxlen=512):
        super(Data_V10, self).__init__()
        nltk.download('punkt')
        self.src = []
        self.dst = []
        self.src_maxlen = maxlen
        self.dst_maxlen = maxlen
        self.tokenizer_src = {'unk':0, 'pad':1, 'bos':2, 'eos':3}
        self.tokenizer_dst = {'unk':0, 'pad':1, 'bos':2, 'eos':3}
        with open(f'./data/CoNaLa/{split}.csv', 'r') as f:
            rdr = csv.reader(f)
            self.tokenizer_src_idx = 4
            self.tokenizer_dst_idx = 4
            for idx, ln in enumerate(rdr):
                if idx > 0:
                    self.src.append(ln[0])
                    self.dst.append(ln[1])                
                    self.make_tokenizer('src', ln[0])
                    self.make_tokenizer('dst', ln[1])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_token = self.tokenization('src', self.src[idx])
        dst_token = self.tokenization('dst', self.dst[idx])
        src_trunc = self.truncation('src', src_token)
        dst_trunc = self.truncation('dst', dst_token)
        src_pad = self.padding('src', src_trunc)
        dst_pad = self.padding('dst', dst_trunc)
        src_out = torch.tensor(src_pad, dtype=torch.long)
        dst_out = torch.tensor(dst_pad, dtype=torch.long)
        return src_out, dst_out

    def make_tokenizer(self, split, sentence):
        tmp = word_tokenize(sentence)
        for word in tmp:
            if split == 'src':
                if not word in self.tokenizer_src:
                    self.tokenizer_src[word] = self.tokenizer_src_idx
                    self.tokenizer_src_idx += 1
            else:
                if not word in self.tokenizer_dst:
                    self.tokenizer_dst[word] = self.tokenizer_dst_idx
                    self.tokenizer_dst_idx += 1

    def tokenization(self, split, sentence):
        token = []
        tmp = word_tokenize(sentence)
        for word in tmp:
            if split == 'src':
                if word in self.tokenizer_src:
                    token.append(self.tokenizer_src[word])
                else:
                    token.append(self.tokenizer_src['unk'])
            else:
                if word in self.tokenizer_dst:
                    token.append(self.tokenizer_dst[word])
                else:
                    token.append(self.tokenizer_dst['unk'])
        
        return token
    
    def truncation(self, split, token):
        if split == 'src':
            return token[:min(self.src_maxlen, len(token))]
        else:
            return token[:min(self.dst_maxlen, len(token))]

    def padding(self, split, token):
        if split == 'src':
            for idx in range(self.src_maxlen - len(token)):
                token.append(self.tokenizer_src['pad'])
        else:
            for idx in range(self.dst_maxlen - len(token)):
                token.append(self.tokenizer_dst['pad'])
        return token