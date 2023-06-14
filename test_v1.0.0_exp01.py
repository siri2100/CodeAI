import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from src import eval


DEVICE     = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PATH_MODEL = './models/v1.0.0_exp01'


def preprocess_v10(batch):
    inputs = tokenizer(batch["intent"], padding="max_length", truncation=True, return_tensors="pt")    
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_code"] = output_str
    return batch


test_df   = pd.read_csv('./data/CoNaLa_tmp/test.csv', delimiter=',', quotechar= '"')
testset   = Dataset.from_pandas(test_df)
model     = AutoModelForSeq2SeqLM.from_pretrained(PATH_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PATH_MODEL, use_fast=True)
evaluator = eval.CodeGenerationEvaluator(tokenizer, DEVICE, smooth_bleu=True)
result    = testset.map(preprocess_v10, batched=True, batch_size=1)