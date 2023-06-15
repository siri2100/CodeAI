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


def scoring(results):
    score_bleu = []
    score_rouge = []
    score_sacrebleu = []
    for i in range(len(results)):
        ref = results["snippet"][i]
        pred = results["pred_code"][i].replace('_',' ').strip()
        if pred is not None and pred != "":
            if ref is not None and ref != "":
                bleu_metric = evaluator.evaluate([pred], [ref])
                score_bleu.append(bleu_metric['BLEU'])
                score_rouge.append(bleu_metric['ROUGE-L'])
                score_sacrebleu.append(bleu_metric['SacreBLEU'])
        else:
            continue
    return score_bleu, score_rouge, score_sacrebleu
                

test_df   = pd.read_csv('./data/CoNaLa/test.csv', delimiter=',', quotechar= '"')
testset   = Dataset.from_pandas(test_df)
model     = AutoModelForSeq2SeqLM.from_pretrained(PATH_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PATH_MODEL, use_fast=True)
evaluator = eval.CodeGenerationEvaluator(tokenizer, DEVICE, smooth_bleu=True)
result    = testset.map(preprocess_v10, batched=True, batch_size=1)
score_bleu, score_rouge, score_sacrebleu = scoring(result)

df = pd.DataFrame({'BLEU':[sum(score_bleu)/len(score_bleu)],
                   'ROUGE Score':[sum(score_rouge)/len(score_rouge)],
                   'Sacre BLEU':[sum(score_sacrebleu)/len(score_sacrebleu)]})
df.to_csv(f'{PATH_MODEL}/result.csv')

print(f'BLEU : {sum(score_bleu)/len(score_bleu)}')
print(f'ROUGE Score : {sum(score_rouge)/len(score_rouge)}')
print(f'Sacre BLEU : {sum(score_sacrebleu)/len(score_sacrebleu)}')