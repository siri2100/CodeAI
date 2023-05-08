import csv

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

''' TODO
    01. NLP에서 data augmentation이 없는가?
    02. class CoNaLa가 완성되면 다른 데이터셋도 동일한 구조로 복사하기(Django, etc)
'''


class CoNaLa(Dataset):
    def __init__(self, file_name):
        super().__init__()
        self.natural_language = []
        self.code = []

        file = open(f"./data/CoNaLa/{file_name}", 'r')
        rdr = csv.reader(file)
        for idx, ln in enumerate(rdr):
            if idx > 0:
                self.natural_language.append(ln[0])
                self.code.append(ln[1])

    def __len__(self):
        return len(self.natural_language)

    def __getitem__(self, idx):
        return self.natural_language[idx], self.code[idx]


if __name__ == "__main__":
    trainset = CoNaLa('train.csv')
    valset = CoNaLa('valid.csv')
    testset = CoNaLa('test.csv')
    print(len(trainset))
    print(len(valset))
    print(len(testset))