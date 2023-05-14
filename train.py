import torch
import torch.nn as nn

from src.data import *
from src.model import *


TRAINSET = 'CoNaLa' # CoNaLa, CoNaLa-Large, Django

''' TODO
    00. Setup Desktop
    01. 추후 MultiGPU 사용이 필요할 시 class MultiGPU 추가
'''

class SingleGPU(nn.Module):
    def __init__(self):
        super().__init__()
        # 01. Dataset
        data_train = CoNaLa('train.csv')
        data_valid = CoNaLa('valid.csv')
        self.train_loader = DataLoader(data_train)
        self.valid_loader = DataLoader(data_valid)

        # 02. Model
        self.model = CodeAI_v1_0()

    def forward(self):
        for idx, (natural_language, code) in enumerate(self.train_loader):
            print('train loop')

        for idx, (natural_language, code) in enumerate(self.valid_loader):
            print('valid loop')


if __name__ == "__main__":
    train = SingleGPU()
