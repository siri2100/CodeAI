import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from src.data import map_to_encoder_decoder_inputs
from src.evaluator import CodeGenerationEvaluator


BATCH_SIZE  = 1
MAX_ENCODER = 32
MAX_DECODER = 32

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

data = load_dataset("AhmedSSoliman/CoNaLa-Large")
trainset = data["train"].map(map_to_encoder_decoder_inputs, tokenizer=tokenizer, batch=BATCH_SIZE, max_encoder=MAX_ENCODER, max_decoder=MAX_DECODER)
validset = data["validation"].map(map_to_encoder_decoder_inputs, tokenizer=tokenizer, batch=1, max_encoder=MAX_ENCODER, max_decoder=MAX_DECODER)
trainset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
validset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    max_length=MAX_ENCODER,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/v1.0.0_exp01",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size="BATCH_SIZE",
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    do_train=True,
    do_eval=True,
    fp16=False,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    seed=1995,
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=CodeGenerationEvaluator,
    data_collator=data_collator,
    train_dataset=trainset,
    eval_dataset=validset,
)

trainer.train()