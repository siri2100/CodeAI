import csv

from transformers import AutoTokenizer
import sentencepiece as spm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

''' TODO
    01. Tokenizer 구현 : AutoTokenizer 
    02. class CoNaLa가 완성되면 다른 데이터셋도 동일한 구조로 복사하기(Django, etc)
'''

MAX_LENGTH = 512

class CoNaLa(Dataset):
    def __init__(self, file_name):
        super().__init__()
        self.natural_language = []
        self.code = []
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl", use_fast = True)

        file = open(f"./data/CoNaLa/{file_name}", 'r')
        rdr = csv.reader(file)
        for idx, ln in enumerate(rdr):
            if idx > 0:
                self.natural_language.append(ln[0])
                self.code.append(ln[1])

    def __len__(self):
        return len(self.natural_language)

    def __getitem__(self, idx):
        # Step 1. Tokenization
        x = self.tokenizer(self.natural_language[idx],
                           max_length=MAX_LENGTH,
                           padding = 'max_length',
                           truncation=True,
                           return_attention_mask = True)
        with self.tokenizer.as_target_tokenizer():
            y = self.tokenizer(self.code[idx],
                               max_length=MAX_LENGTH,
                               padding='max_length',
                               truncation=True,
                               return_attention_mask=True)
        

        return x, y


if __name__ == "__main__":
    trainset = CoNaLa('train.csv')
    valset = CoNaLa('valid.csv')
    testset = CoNaLa('test.csv')
    print(len(trainset))
    print(len(valset))
    print(len(testset))