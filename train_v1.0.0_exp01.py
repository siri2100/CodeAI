import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from src.evaluator import CodeGenerationEvaluator


BATCH_SIZE  = 128
DEVICE      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCH       = 10
LR          = 1e-5
MAX_ENCODER = 8
MAX_DECODER = 8


def map_to_encoder_decoder_inputs(batch):    
    inputs = tokenizer(batch["intent"], padding="max_length", truncation=True, max_length=MAX_ENCODER)
    outputs = tokenizer(batch["snippet"], padding="max_length", truncation=True, max_length=MAX_DECODER)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask
    return batch

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
data = load_dataset("AhmedSSoliman/CoNaLa-Large")
trainset = data["train"].map(map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['intent', 'snippet'])
validset = data["validation"].map(map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['intent', 'snippet'])
trainset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
validset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    max_length=MAX_ENCODER,
    padding='max_length',
    model=model,
)
evaluator = CodeGenerationEvaluator(tokenizer, DEVICE, smooth_bleu=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/v1.0.0_exp01",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
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
