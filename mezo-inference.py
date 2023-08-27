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
    evaluate,
    CustomTrainer
)

from vars import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_map = {"": 0}

#infer_peft_config = PeftConfig.from_pretrained('bits-ft-C-I-R')
infer_peft_config = PeftConfig.from_pretrained('bits-ft')

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

tokenizer = AutoTokenizer.from_pretrained('bits-ft-I-R')

print("Special tokens:", tokenizer.special_tokens_map)
print("BOS Token:", tokenizer.bos_token)
print("EOS Token:", tokenizer.eos_token)
print("PAD Token:", tokenizer.pad_token)
print("UNK Token:", tokenizer.unk_token)
print("SEP Token:", tokenizer.sep_token)
print("CLS Token:", tokenizer.cls_token)
print("MASK Token:", tokenizer.mask_token)

query_text = (
f"""
Context:

Until 1917, it was possible for someone who was not a priest, but only in minor orders, to become a cardinal (see "lay cardinals", below), but they were enrolled only in the order of cardinal deacons. For example, in the 16th century, Reginald Pole was a cardinal for 18 years before he was ordained a priest. In 1917 it was established that all cardinals, even cardinal deacons, had to be priests, and, in 1962, Pope John XXIII set the norm that all cardinals be ordained as bishops, even if they are only priests at the time of appointment. As a consequence of these two changes, canon 351 of the 1983 Code of Canon Law requires that a cardinal be at least in the order of priesthood at his appointment, and that those who are not already bishops must receive episcopal consecration. Several cardinals aged over 80 or close to it when appointed have obtained dispensation from the rule of having to be a bishop. These were all appointed cardinal-deacons, but one of them, Roberto Tucci, lived long enough to exercise the right of option and be promoted to the rank of cardinal-priest.

Prompt:

TL;DR

Response:
"""
)
torch.manual_seed(SEED)

# attention_mask = torch.ones_like(input_ids)
generator = pipeline('text-generation', model=infer_model, tokenizer=tokenizer,
                     min_length=50,
                     max_length=512,
                     temperature=1,
                     # attention_mask=attention_mask,
                     do_sample=True,
                     top_k=50,
                     top_p=1,
                     num_return_sequences=1,
                     no_repeat_ngram_size=2,
                     num_beams=5,
                     early_stopping=True)

# results = generator(query_text, do_sample=True, min_length=50, max_length=200)
results = generator(query_text)
print(results[0]['generated_text'])
