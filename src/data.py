from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k


''' TODO
    01. 추후 Multi30k 벤치마킹하여 CoNaLa 구현 (아래 코드 참고)
'''

''' CoNaLa
    MAX_LEN = 512
    class CoNaLa(Dataset):
    def __init__(self, file_name):
        super().__init__()
        self.nl = []
        self.code = []
        self.tokenizer = AutoTokenizer.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa-Large")

        file = open(f"./data/CoNaLa/{file_name}", 'r')
        rdr = csv.reader(file)
        for idx, ln in enumerate(rdr):
            if idx > 0:
                self.nl.append(ln[0])
                self.code.append(ln[1])

    def __len__(self):
        return len(self.nl)

    def __getitem__(self, idx):
        # Step 1. Tokenization
        # x = self.tokenizer(self.nl[idx], padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")       
        # y = self.tokenizer(self.code[idx], padding='max_length', truncation=True, max_length=MAX_LEN, return_tensor="pt")
        x = self.nl[idx]
        y = self.code[idx]
        return x, y
'''

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

# 토큰들이 어휘집(vocab)에 인덱스 순서대로 잘 삽입되어 있는지 확인합니다.
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

text_transform = {}
token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# 토큰 목록을 생성하기 위한 헬퍼 함수
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # 학습용 데이터 반복자
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # torchtext의 vocab 객체 생성
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# 'UNK_IDX'를 기본 인덱스로 설정합니다.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, dst, device):
    src_mask = torch.zeros((src.shape[1], src.shape[1]), device=device).type(torch.bool)    # 27 x 27
    dst_mask = generate_square_subsequent_mask(dst.shape[1], device)                        # 23 x 23
    src_padding_mask = (src == PAD_IDX)                                                     # PAD_IDX = 1, 128 x 27
    dst_padding_mask = (dst == PAD_IDX)                                                     # 128 x 23
    return src_mask, dst_mask, src_padding_mask, dst_padding_mask


class Data_V10(nn.Module):
    def __init__(self, split='train', maxlen=512):  
        super(Data_V10, self).__init__()
        # 01. Load
        self.src = []
        self.dst = []
        self.src_maxlen = maxlen
        self.dst_maxlen = maxlen
        with open(f'./data/Multi30k/{split}_en.csv', 'r') as f:
            while True:
                ln = f.readline().rstrip()
                if not ln: break
                self.src.append(ln)
        with open(f'./data/Multi30k/{split}_en.csv', 'r') as f:
            while True:
                ln = f.readline().rstrip()
                if not ln: break
                self.dst.append(ln)
        
        # 02. Preprocess
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        self.token_transform = get_tokenizer('spacy', language='en_core_web_sm') # 토큰화(Tokenization)
        self.vocab_transform = build_vocab_from_iterator(yield_tokens(train_iter, 'en'), min_freq=1, specials=special_symbols, special_first=True) # 수치화(Numericalization)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_token = self.token_transform(self.src[idx])
        dst_token = self.token_transform(self.dst[idx])
        src_vocab = torch.tensor(self.vocab_transform(src_token))
        dst_vocab = torch.tensor(self.vocab_transform(dst_token))
        src_pad = F.pad(src_vocab, (0, self.src_maxlen-len(src_vocab)))
        dst_pad = F.pad(dst_vocab, (0, self.dst_maxlen-len(dst_vocab)))
        return src_pad, dst_pad

    def tensor_transform(token_ids: List[int]):
        # BOS/EOS를 추가하고 입력 순서(sequence) 인덱스에 대한 텐서를 생성하는 함수
        return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

