import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from src.eval import CodeGenerationEvaluator


BATCH_SIZE  = 8
DEVICE      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCH       = 30
LR          = 1e-5
MAX_LEN     = 512
PATH_OUTPUT = "./models/v1.0.0_exp01"


def preprocess_v10(examples):
    model_inputs = tokenizer(examples['intent'], max_length=MAX_LEN, padding = 'max_length', truncation=True, return_attention_mask = True)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['snippet'], max_length=MAX_LEN, padding='max_length', truncation=True, return_attention_mask=True)
    model_inputs['labels'] = targets['input_ids']
    model_inputs['decoder_input_ids'] = targets['input_ids']
    model_inputs['decoder_attention_mask'] = targets['attention_mask']
    return model_inputs


train_df    = pd.read_csv('./data/CoNaLa/train.csv', delimiter=',', quotechar= '"')
valid_df    = pd.read_csv('./data/CoNaLa/valid.csv', delimiter=',', quotechar= '"')
trainset    = Dataset.from_pandas(train_df)
validset    = Dataset.from_pandas(valid_df)
tokenizer   = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl", use_fast=True)
model       = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
evaluator   = CodeGenerationEvaluator(tokenizer, DEVICE, smooth_bleu=True)
data_col    = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=MAX_LEN, model=model)
train_input = trainset.map(preprocess_v10, batched=True)
valid_input = validset.map(preprocess_v10, batched=True)
train_input = train_input.remove_columns(['intent', 'snippet'])
valid_input = valid_input.remove_columns(['intent', 'snippet'])

training_args = Seq2SeqTrainingArguments(
    output_dir=PATH_OUTPUT,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCH,
    do_train=True,
    do_eval=False,
    fp16=False,
    overwrite_output_dir=True,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.05,
    seed=1,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_col,
    train_dataset=train_input,
    eval_dataset=valid_input,
)

trainer.train()
trainer.save_model()
trainer.save_state()
