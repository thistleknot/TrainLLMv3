#!/usr/bin/env python
# coding: utf-8

from common_imports import *
from functions import (
    create_subset,
    filter_datasets_for_use_case,
    split_datasets,
    unique_elements,
    PerplexityLoggingCallback,
    print_trainable_parameters,
    CustomDataset,
    get_sequences,
    CustomTrainer
)

from vars import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_map = {"": 0}

#infer_peft_config = PeftConfig.from_pretrained('bits-ft-C-I-R')
infer_peft_config = PeftConfig.from_pretrained('bits')

infer_model = AutoModelForCausalLM.from_pretrained(
    infer_peft_config.base_model_name_or_path,
    quantization_config=bnb_config, device_map=device_map
)

# enabling raises memory requirements, but much faster inference
infer_model.gradient_checkpointing_enable()
infer_model = prepare_model_for_kbit_training(infer_model)
infer_model = get_peft_model(infer_model, lora_config)
print_trainable_parameters(infer_model)
# only necessary when use with the above
infer_model.model.config.use_cache = True
infer_model.config.use_cache = True

#tokenizer = AutoTokenizer.from_pretrained('bits-ft')

tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False)

tokenizer.add_special_tokens(special_tokens_dict)

infer_model.resize_token_embeddings(len(tokenizer))

print("Special tokens:", tokenizer.special_tokens_map)
print("BOS Token:", tokenizer.bos_token)
print("EOS Token:", tokenizer.eos_token)
print("PAD Token:", tokenizer.pad_token)
print("UNK Token:", tokenizer.unk_token)
print("SEP Token:", tokenizer.sep_token)
print("CLS Token:", tokenizer.cls_token)
print("MASK Token:", tokenizer.mask_token)

query_text = (
f"""Instruction:

What are all the different ways of saying customer lifetime value?

Answer:

Customer lifetime value is commonly referred""")
torch.manual_seed(SEED)

# attention_mask = torch.ones_like(input_ids)
generator = pipeline('text-generation', model=infer_model, tokenizer=tokenizer,
                     min_length=50,
                     max_length=512,
                     temperature=0.7,
                     # attention_mask=attention_mask,
                     do_sample=True,
                     top_k=50,
                     top_p=0.9,
                     num_return_sequences=1,
                     no_repeat_ngram_size=2,
                     #num_beams=5,
                     #early_stopping=True
)

# results = generator(query_text, do_sample=True, min_length=50, max_length=200)
results = generator(query_text)
print(results[0]['generated_text'])
