import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import *
from src.model import *


EXP_NAME      = 'v1.0'
TRAINSET      = 'CoNaLa' # CoNaLa, CoNaLa-Large, Django
EPOCH         = 2
BATCH_SIZE    = 3
LEARNING_RATE = 1e-4
DEVICE        = 'cuda:0'


'''VERSION
    CUDA        : 11.8
    PyTorch     : 2.0.0
    Python      : 3.9.16
    model 이름  : nato phonetic alphabet순으로 정함 (추후 정리)
'''

''' TODO
    01. Debugging
        문제 : 258 iteration에서 에러 발생
'''

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 2
BATCH_SIZE = 8 # 16

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
MAX_LEN = 500
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512


class SingleGPU(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(f'./models/{EXP_NAME}', exist_ok=True)

        data_train = Data_V10(split='train', maxlen=MAX_LEN)
        data_valid = Data_V10(split='valid', maxlen=MAX_LEN)
        self.train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, num_workers=1)
        self.valid_loader = DataLoader(data_valid, batch_size=BATCH_SIZE, num_workers=1)
        
        self.model = Model_V10_Alpha(
            DEVICE,
            NUM_DECODER_LAYERS,
            NUM_ENCODER_LAYERS,
            EMB_SIZE,
            NHEAD,
            len(data_train.tokenizer_src),
            len(data_train.tokenizer_dst),
            FFN_HID_DIM
        )
        self.model = self.model.to(DEVICE)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)

    def forward(self, epoch):
        # 01. Train
        self.model.train()
        for idx, (src, dst) in enumerate(self.train_loader):
            # 01-0. Preprocess
            src, dst = src.to(DEVICE), dst.to(DEVICE)

            # 01-1. Forward Propagation
            dst_pd = self.model(src, dst)

            # 01-2. Backward Propagation
            self.optimizer.zero_grad()
            loss = self.loss_fn(dst_pd.reshape(-1, dst_pd.shape[-1]), dst.reshape(-1))
            loss.backward()
            self.optimizer.step()

            print(f'Train Loop || Epoch : {epoch+1}, Iteration : {idx + 1} / {len(self.train_loader)}, Loss : {loss}')

        # 01-3. Save
        epoch_num = str(epoch + 1).rjust(3, '0')
        torch.save(self.model.state_dict(), f'./models/{EXP_NAME}/epoch_{epoch_num}.pth')

        # 02. Valid
        with torch.no_grad():
            self.model.eval()
            for idx, (src, dst) in enumerate(self.valid_loader):
                # 01-0. Preprocess
                src, dst = src.to(DEVICE), dst.to(DEVICE)

                # 01-1. Forward Propagation
                dst_pd = self.model(src, dst)

                # 01-2. Backward Propagation
                loss = self.loss_fn(dst_pd.reshape(-1, dst_pd.shape[-1]), dst.reshape(-1))

                print(f'Valid Loop || Epoch : {epoch+1}, Iteration : {idx + 1} / {len(self.valid_loader)}, Loss : {loss}')


if __name__ == "__main__":
    train = SingleGPU()
    for epoch in range(EPOCH):
        train(epoch)
