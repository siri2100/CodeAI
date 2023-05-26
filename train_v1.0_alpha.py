import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import *
from src.model import *


EXP_NAME      = 'v1.0.0'
TRAINSET      = 'CoNaLa' # CoNaLa, CoNaLa-Large, Django
EPOCH         = 2
BATCH_SIZE    = 2
LEARNING_RATE = 1e-4
DEVICE        = 'cpu'


'''VERSION
    CUDA        : 11.8
    PyTorch     : 2.0.0
    Python      : 3.9.16
    model 이름  : nato phonetic alphabet순으로 정함 (추후 정리)
'''

''' TODO
    01. Multi30k예제 따라하고 있음 (현재 dataset loading부분)
'''

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 2
BATCH_SIZE = 128

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 512
NHEAD = 8
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
FFN_HID_DIM = 512


class SingleGPU(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(f'./models/{EXP_NAME}', exist_ok=True)

        data_train = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        data_valid = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        self.train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=1)
        self.valid_loader = DataLoader(data_valid, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=1)
        
        self.model = Model_V10_Alpha(
            NUM_DECODER_LAYERS,
            NUM_ENCODER_LAYERS,
            EMB_SIZE,
            NHEAD,
            SRC_VOCAB_SIZE,
            TGT_VOCAB_SIZE,
            FFN_HID_DIM
        )
        self.model = self.model.to(DEVICE)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)

    def forward(self, epoch):
        # 01. Train
        self.model.train()
        for idx, (src, tgt) in enumerate(self.train_loader):
            print(f'Train Loop || Epoch : {epoch+1}, Iteration : {idx + 1}')
            # 01-0. Preprocess
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1], DEVICE)

            # 01-1. Forward Propagation
            code_pd = self.model(src, tgt)

            # 01-2. Backward Propagation
            # self.optimizer.zero_grad()
            # loss = self.loss(src[0], tgt)
            # loss.backward()
            # self.optimizer.step()

        # 01-3. Save
        epoch_num = str(epoch).rjust(3, '0')
        torch.save(self.model.state_dict(), f'./models/{EXP_NAME}/epoch_{epoch_num}.pth')

        # 02. Valid
        self.model.eval()
        for idx, (src, tgt) in enumerate(self.valid_loader):
            print(f'Valid Loop || Epoch : {epoch+1}, Iteration : {idx + 1}/{len(self.valid_loader)}')

            # 02-1. Forward Propagation
            code_pd = self.model(src)


if __name__ == "__main__":
    train = SingleGPU()
    for epoch in range(EPOCH):
        train(epoch)
