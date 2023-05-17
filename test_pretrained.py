from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa-Large")
tokenizer = AutoTokenizer.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa-Large")


NL_input = "send a signal `signal.sigusr1` to the current process"
while True:
    output = model.generate(**tokenizer(NL_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt"))
    output_code = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)
    print(output_code)