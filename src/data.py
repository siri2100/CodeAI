import csv

from transformers import AutoTokenizer
import sentencepiece as spm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

''' TODO
    01. 추후 MarianTokenizer 구현 (아래 코드 참고)

        MarianTokenizer(name_or_path='AhmedSSoliman/MarianCG-CoNaLa-Large', vocab_size=67028, model_max_length=512, 
        is_fast=False, padding_side='right', truncation_side='right', special_tokens={'eos_token': '</s>', 
        'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=True)
'''

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
        x = self.tokenizer(self.nl[idx], padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")       
        y = self.tokenizer(self.code[idx], padding='max_length', truncation=True, max_length=MAX_LEN, return_tensor="pt")
        return x, y


if __name__ == "__main__":
    trainset = CoNaLa('train.csv')
    valset = CoNaLa('valid.csv')
    testset = CoNaLa('test.csv')
    print(len(trainset))
    print(len(valset))
    print(len(testset))