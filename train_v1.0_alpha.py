import os
import sys

import torch
import torch.nn as nn

from src.data import *
from src.model import *


EXP_NAME      = 'v1.0.0'
TRAINSET      = 'CoNaLa' # CoNaLa, CoNaLa-Large, Django
EPOCH         = 2
LEARNING_RATE = 1e-4


'''VERSION
    CUDA        : 11.8
    PyTorch     : 2.0.0
    Python      : 3.9.16
    model 이름  : nato phonetic alphabet순으로 정함 (추후 정리)
'''

''' TODO
    01. model.py : MarianCG Model 구현
    02. v1.0_alpha 학습 및 평가하기
'''


class SingleGPU(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(f'./models/{EXP_NAME}', exist_ok=True)

        data_train = CoNaLa('train.csv')
        data_valid = CoNaLa('valid.csv')
        self.train_loader = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=1)
        self.valid_loader = DataLoader(data_valid, batch_size=1, shuffle=False, num_workers=1)

        self.model = CodeAI_V10_Alpha()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.L1Loss()
        
    def forward(self, epoch):
        # 01. Train
        self.model.train()
        for idx, (nl, code_gt) in enumerate(self.train_loader):
            print(f'Train Loop || Epoch : {epoch+1}, Iteration : {idx + 1}/{len(self.train_loader)}')

            # 01-1. Forward Propagation
            code_pd = self.model(nl[0])

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