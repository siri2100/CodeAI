import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from src.evaluator import CodeGenerationEvaluator


BATCH_SIZE  = 4
DEVICE      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCH       = 10
LR          = 1e-5


def preprocess(examples,MAX_LENGTH = 512):
    model_inputs = tokenizer(examples['intent'],
                             max_length=MAX_LENGTH,
                             padding = 'max_length',
                             truncation=True,
                             return_attention_mask = True)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['snippet'],
                            max_length=MAX_LENGTH,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True)
    model_inputs['labels'] = targets['input_ids']
    model_inputs['decoder_input_ids'] = targets['input_ids']
    model_inputs['decoder_attention_mask'] = targets['attention_mask']
    return model_inputs


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

train_df = pd.read_csv('./data/CoNaLa/train.csv',delimiter=',', quotechar= '"')
val_df = pd.read_csv('./data/CoNaLa/valid.csv',delimiter=',', quotechar= '"')
test_df = pd.read_csv('./data/CoNaLa/test.csv',delimiter=',', quotechar= '"')
trainset = Dataset.from_pandas(train_df)
validset = Dataset.from_pandas(val_df)
testset = Dataset.from_pandas(test_df)
trainset = trainset.map(preprocess, batched=True)
trainset = trainset.remove_columns(['intent', 'snippet'])
validset = validset.map(preprocess, batched=True)
validset = validset.remove_columns(['intent', 'snippet'])

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    max_length=512,
    model=model,
)
evaluator = CodeGenerationEvaluator(tokenizer, DEVICE, smooth_bleu=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/v1.0.0_exp01",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCH,
    do_train=True,
    do_eval=True,
    fp16=False,
    overwrite_output_dir=True,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.05,
    seed=1995,
    load_best_model_at_end=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=evaluator,
    data_collator=data_collator,
    train_dataset=trainset,
    eval_dataset=validset,
)

trainer.train()
