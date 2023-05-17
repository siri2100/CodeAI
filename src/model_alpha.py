import torch
import torch.nn as nn


''' TODO
    ** model 이름은 nato phonetic alphabet순으로 정함 (추후 정리)
    01. CodeAI_v1_0 코드 작성 (test_pretrained의 tokenizer랑 model source code 참고해서 작성)
'''


class CodeAI_v1_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 20)
    
    def forward(self, x):
        return x