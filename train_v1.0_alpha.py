import csv
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import *
from src.model import *


''' TODO
    Training이 끝난 후 loss 및 성능 분석 수행 -> 다음 방향 정하기
'''

DEVICE                      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_EXP_NAME              = 'v1.0_Alpha'
TRAIN_EPOCH                 = 100
TRAIN_LEARNING_RATE         = 1e-4
TRAIN_BATCH_SIZE            = 8 # 16
DATA_TRAINSET               = 'CoNaLa' # CoNaLa, CoNaLa-Large, Django
DATA_MAX_LEN                = 500
MODEL_NUM_ENCODER_LAYERS    = 3
MODEL_NUM_DECODER_LAYERS    = 3
MODEL_EMB_SIZE              = 512
MODEL_NHEAD                 = 8
MODEL_FFN_HID_DIM           = 512


class SingleGPU(nn.Module):
    def __init__(self):
        super().__init__()
        os.makedirs(f'./models/{TRAIN_EXP_NAME}', exist_ok=True)

        # 01. Set Dataset
        data_train = Data_V10(split='train', maxlen=DATA_MAX_LEN)
        data_valid = Data_V10(split='valid', maxlen=DATA_MAX_LEN)
        self.train_loader = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, num_workers=1)
        self.valid_loader = DataLoader(data_valid, batch_size=TRAIN_BATCH_SIZE, num_workers=1)

        # 02. Set Model
        self.model = Model_V10_Alpha(
            DEVICE,
            MODEL_NUM_DECODER_LAYERS,
            MODEL_NUM_ENCODER_LAYERS,
            MODEL_EMB_SIZE,
            MODEL_NHEAD,
            len(data_train.tokenizer_src),
            len(data_train.tokenizer_dst),
            MODEL_FFN_HID_DIM
        )
        self.model = self.model.to(DEVICE)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 03. Set Loss & Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=TRAIN_LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)

        # 04. Set Loss log & Hyperparameter
        with open(f'./models/{TRAIN_EXP_NAME}/train_loss.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['Step', 'Loss'])
        with open(f'./models/{TRAIN_EXP_NAME}/valid_loss.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['Step', 'Loss'])
        with open(f'./models/{TRAIN_EXP_NAME}/hyperparam.txt', 'w', newline='') as f:
            f.write(f'TRAIN_EXP_NAME : {TRAIN_EXP_NAME}\n')
            f.write(f'TRAIN_EPOCH : {TRAIN_EPOCH}\n')
            f.write(f'TRAIN_LEARNING_RATE : {TRAIN_LEARNING_RATE}\n')
            f.write(f'TRAIN_BATCH_SIZE : {TRAIN_BATCH_SIZE}\n\n')
            f.write(f'DATA_TRAINSET : {DATA_TRAINSET}\n')
            f.write(f'DATA_MAX_LEN : {DATA_MAX_LEN}\n\n')
            f.write(f'MODEL_NUM_ENCODER_LAYERS : {MODEL_NUM_ENCODER_LAYERS}\n')
            f.write(f'MODEL_NUM_DECODER_LAYERS : {MODEL_NUM_DECODER_LAYERS}\n')
            f.write(f'MODEL_EMB_SIZE : {MODEL_EMB_SIZE}\n')
            f.write(f'MODEL_NHEAD : {MODEL_NHEAD}\n')
            f.write(f'MODEL_FFN_HID_DIM : {MODEL_FFN_HID_DIM}\n')

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

            # 01-3. Print & Save Loss
            print(f'Train Loop || Epoch : {epoch+1}, Iteration : {idx + 1} / {len(self.train_loader)}, Loss : {loss}')
            with open(f'./models/{TRAIN_EXP_NAME}/train_loss.csv', 'a', newline='') as f:
                wr = csv.writer(f)
                wr.writerow([epoch*len(self.train_loader) + idx, loss])

        # 01-4. Save Model
        epoch_num = str(epoch + 1).rjust(3, '0')
        torch.save(self.model.state_dict(), f'./models/{TRAIN_EXP_NAME}/epoch_{epoch_num}.pth')
        
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

                # 01-3. Print & Save Loss
                print(f'Valid Loop || Epoch : {epoch+1}, Iteration : {idx + 1} / {len(self.valid_loader)}, Loss : {loss}')
                with open(f'./models/{TRAIN_EXP_NAME}/valid_loss.csv', 'a', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow([(epoch+1)*len(self.train_loader)-1, loss])


if __name__ == "__main__":
    train = SingleGPU()
    for epoch in range(TRAIN_EPOCH):
        train(epoch)
