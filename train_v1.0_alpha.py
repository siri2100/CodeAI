import os
import sys

import torch
import torch.nn as nn

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
    01. v1.0_alpha(Transformer) 파라미터 설정
    02. Transformer 공부
'''


class SingleGPU(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(f'./models/{EXP_NAME}', exist_ok=True)

        data_train = CoNaLa('train.csv')
        data_valid = CoNaLa('valid.csv')
        self.train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.valid_loader = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        self.model = Model_V10_Alpha(
            src_vocab_size=10,
            tgt_vocab_size=10,
            device=DEVICE,
            max_len=512,
            d_embed=10,
            n_layer=1,
            d_model=10,
            h=10,
            d_ff=2048,
            drop_rate=0.1,
            norm_eps=1e-5,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.L1Loss()
        
    def forward(self, epoch):
        # 01. Train
        self.model.train()
        for idx, (nl, code_gt) in enumerate(self.train_loader):
            print(f'Train Loop || Epoch : {epoch+1}, Iteration : {idx + 1}/{len(self.train_loader)}')

            # 01-1. Forward Propagation
            code_pd = self.model(nl, code_gt)

            # 01-2. Backward Propagation
            # self.optimizer.zero_grad()
            # loss = self.loss(code_gt[0], code_pd)
            # loss.backward()
            # self.optimizer.step()

        # 01-3. Save
        epoch_num = str(epoch).rjust(3, '0')
        torch.save(self.model.state_dict(), f'./models/{EXP_NAME}/epoch_{epoch_num}.pth')

        # 02. Valid
        self.model.eval()
        for idx, (nl, code_gt) in enumerate(self.valid_loader):
            print(f'Valid Loop || Epoch : {epoch+1}, Iteration : {idx + 1}/{len(self.valid_loader)}')

            # 02-1. Forward Propagation
            code_pd = self.model(nl[0])


if __name__ == "__main__":
    train = SingleGPU()
    for epoch in range(EPOCH):
        train(epoch)
