from typing import Iterable, List

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
# 특수 기호와 인덱스를 정의함
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# 토큰들이 어휘집(vocab)에 인덱스 순서대로 잘 삽입되어 있는지 확인합니다.
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

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


# 데이터를 텐서로 조합(collate)하는 함수
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    tgt_mask = generate_square_subsequent_mask(tgt.shape[0])                                # 23 x 23
    src_mask = torch.zeros((src.shape[0], src.shape[0]),device=DEVICE).type(torch.bool)     # 27 x 27
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)                                     # PAD_IDX = 1, 128 x 27
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)                                     # 128 x 23
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    trainset = CoNaLa('train.csv')
    valset = CoNaLa('valid.csv')
    testset = CoNaLa('test.csv')
    print(len(trainset))
    print(len(valset))
    print(len(testset))