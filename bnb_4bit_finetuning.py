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


tokenizer = AutoTokenizer.from_pretrained(args['model_id'])

peft_config = PeftConfig.from_pretrained('bits')

model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config, device_map=device_map
)


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print_trainable_parameters(model)

#peft wrapper feature
model.config.use_cache = False

tokenizer.pad_token = tokenizer.eos_token

print("before load")
with open('../venv_train_neo/datasets_dict.pkl', 'rb') as f:
    datasets_dict = pickle.load(f)
    
finetune_datasets = filter_datasets_for_use_case(datasets_dict, 'finetune')
train_data_list, valid_data_list, valid_data_indices = split_datasets(finetune_datasets, ratio=args['split_ratio'], random_state=args['seed'])

train_data_list = [record for dataset in train_data_list.values() for record in dataset]
valid_data_list = [record for dataset in valid_data_list.values() for record in dataset]

combined_train = tokenizer.eos_token.join(train_data_list)
combined_valid = tokenizer.eos_token.join(valid_data_list)

train_sequences = get_sequences(combined_train, tokenizer,seq_length=args['seq_length'])
print(combined_train.count(tokenizer.eos_token)*2)
print(np.sum([t.count(tokenizer.eos_token_id) for t in train_sequences]))

valid_sequences = get_sequences(combined_valid, tokenizer, seq_length=args['seq_length'])

train_epoch_steps = (len(train_sequences) / (args['batch_size'] * args['gradient_accumulation_steps']))
valid_epoch_steps = (len(valid_sequences) / (args['batch_size'] * args['gradient_accumulation_steps']))

max_train_steps = int(train_epoch_steps * args['epochs'])

train_dataset = datasets.Dataset.from_dict({"input_ids": train_sequences})
valid_dataset = datasets.Dataset.from_dict({"input_ids": valid_sequences})

subset_valid_dataset = create_subset(valid_dataset, args['num_eval_examples'])

#trainer = Trainer(
trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=subset_valid_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size = args['batch_size'],
        per_device_eval_batch_size = args['batch_size'],
        gradient_accumulation_steps=args['gradient_accumulation_steps'],
        warmup_steps=int(train_epoch_steps * args['warm_ratio']),
        evaluation_strategy='steps',
        max_steps=max_train_steps,
        learning_rate=args['learning_rate']/10,
        weight_decay=args['weight_decay'],      # Add weight decay argument
        lr_scheduler_type='cosine',     # Add lr scheduler type argument
        max_grad_norm=args['max_grad_norm'],       # Add max norm clipping argument
        fp16=True,  # Add a keyword here
        logging_steps=int(np.clip(np.round(train_epoch_steps/10),1,1)),
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[PerplexityLoggingCallback()],  # Add the custom callback
)

trainer.train()

initial_completed_steps = trainer.get_completed_steps()

valid_steps = max(1, int(np.round((initial_completed_steps/train_epoch_steps * valid_epoch_steps),0)))

valid_trainer = CustomTrainer(
    model=trainer.model,
    max_eval_steps=valid_steps,  # Pass the valid_steps here
    train_dataset=valid_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size = args['batch_size'],
        per_device_eval_batch_size = args['batch_size'],
        gradient_accumulation_steps=args['gradient_accumulation_steps'],
        #warmup_steps=int(train_epoch_steps * args['warm_ratio']),
        #evaluation_strategy='steps',
        max_steps=valid_steps,
        learning_rate=args['learning_rate']/10,
        weight_decay=args['weight_decay'],      # Add weight decay argument
        lr_scheduler_type='cosine',     # Add lr scheduler type argument
        max_grad_norm=args['max_grad_norm'],       # Add max norm clipping argument
        fp16=True,  # Add a keyword here
        logging_steps=int(np.clip(np.round(train_epoch_steps/10),1,1)),
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

valid_trainer.train()

valid_trainer.model.save_pretrained('./bitsft')

