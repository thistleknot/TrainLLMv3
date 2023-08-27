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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("resources/train.json", "r") as f:
    args = json.load(f)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args["load_in_4bit"],
    bnb_4bit_use_double_quant=args["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=args["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=eval(args["bnb_4bit_compute_dtype"])
)

lora_config = LoraConfig(
    r=args["lora_r"],
    lora_alpha=args["lora_alpha"],
    lora_dropout=args["lora_dropout"],
    bias=args["lora_bias"],
    task_type=args["lora_task_type"]
)

device_map = {"": 0}

infer_peft_config = PeftConfig.from_pretrained('bitsft')

infer_model = AutoModelForCausalLM.from_pretrained(
        infer_peft_config.base_model_name_or_path,
        quantization_config=bnb_config, device_map=device_map
)

#enabling raises memory requirements, but much faster inference
infer_model.gradient_checkpointing_enable()
infer_model = prepare_model_for_kbit_training(infer_model)
infer_model = get_peft_model(infer_model, lora_config)
print_trainable_parameters(infer_model)
#only necessary when use with the above
infer_model.model.config.use_cache = True

tokenizer = AutoTokenizer.from_pretrained(args['model_id'])
tokenizer.pad_token = tokenizer.eos_token

query_text = (
    "Context:\n"
    "Jane picked 2 apples and 2 oranges from a nearby grove.\n"
    "Prompt:\n"
    "Let's think step by step.  How many pieces of fruit does Jane have?\nBe sure to explain your reasoning.\n"
    "Response:\n"
)

query_text = (
    "Prompt:\n"
    "What is the meaning of life?\n"
    "Response:\n"
)

#attention_mask = torch.ones_like(input_ids)
generator = pipeline('text-generation', model=infer_model, tokenizer = tokenizer,
    min_length=50,
    max_length=128,
    temperature=1,
    #attention_mask=attention_mask,
    do_sample=True,
    top_k=50,
    top_p=1,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    num_beams=5,
    early_stopping=True)

#results = generator(query_text, do_sample=True, min_length=50, max_length=200)
results = generator(query_text)
print(results[0]['generated_text'])
