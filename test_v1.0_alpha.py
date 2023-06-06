import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import *
from src.model import *


''' TODO
    Training이 끝난 후 loss 및 성능 분석 수행 -> 다음 방향 정하기
'''

DEVICE                      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_EXP_NAME               = 'v1.0_Alpha'
TEST_EPOCH                  = 100
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
        # 01. Set Dataset
        data_test = Data_V10(split='train', maxlen=DATA_MAX_LEN)
        self.test_loader = DataLoader(data_test, batch_size=1, num_workers=1)

        # 02. Set Model
        self.model = Model_V10_Alpha(
            DEVICE,
            MODEL_NUM_DECODER_LAYERS,
            MODEL_NUM_ENCODER_LAYERS,
            MODEL_EMB_SIZE,
            MODEL_NHEAD,
            5888,
            10588,
            MODEL_FFN_HID_DIM
        )
        self.model.load_state_dict(torch.load(f'./models/{TEST_EXP_NAME}/epoch/epoch_{TEST_EPOCH}.pth', map_location=DEVICE))
        self.model.eval()
        self.model = self.model.to(DEVICE)

    def forward(self):
        with torch.no_grad():
            for idx, (src, dst) in enumerate(self.test_loader):
                # 01-0. Preprocess
                src, dst = src.to(DEVICE), dst.to(DEVICE)

                # 01-1. Forward Propagation
                dst_pd = self.model(src, dst)
                dst_pd = dst_pd.reshape(-1, dst_pd.shape[-1])


if __name__ == "__main__":
    test = SingleGPU()
    test()
