#!/usr/bin/env python
# coding: utf-8

#!pip install bitsandbytes git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/accelerate.git datasets torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

###Masked language Modeling
"""
Your model definition looks good as it is already set up for causal language modeling. Since MLM is not natively supported by GPT-NEO models, applying MLM directly to GPT-NEO would require substantial changes to the architecture, and its effectiveness is not guaranteed. However, you can still perform masked language modeling with GPT-2 models or BERT models.

If you want to proceed with GPT-2 or BERT for MLM, follow these steps:

1. Change the `model_id` to a GPT-2 or BERT model, for example:
```python
model_id = "gpt2"
```
or
```python
model_id = "bert-base-uncased"
```

2. Modify the model import statement based on the chosen model. For GPT-2, use:
```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(model_id)
```
For BERT, use:
```python
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained(model_id)
```

3. Update the data collator in your trainer setup to enable masking, by setting `mlm` to `True`:

```python
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=True)
```

4. Use this data collator when initializing your `CustomTrainer`.

Please note that these changes are applicable only if you choose GPT-2 or BERT models for MLM tasks. Since your current model is a GPT-NEO-based model, applying MLM directly might not be straightforward or guaranteed to work well.
"""

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
model = AutoModelForCausalLM.from_pretrained(args['model_id'], quantization_config=bnb_config, device_map=device_map)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)
model.config.use_cache = False

print_trainable_parameters(model)

tokenizer.pad_token = tokenizer.eos_token

print("before load")
with open('../venv_train_neo/datasets_dict.pkl', 'rb') as f:
    datasets_dict = pickle.load(f)
pretrain_datasets = filter_datasets_for_use_case(datasets_dict, 'pretrain')

finetune_datasets = filter_datasets_for_use_case(datasets_dict, 'finetune')

train_data_list, valid_data_list, valid_data_indices = split_datasets(finetune_datasets, ratio=args['split_ratio'], random_state=args['seed'])

# Filter out blank or np.nan values for each dataset in pretrain_train_data
filtered_train_data_list = {}
for key, value in train_data_list.items():
    filtered_train_data_list[key] = [record for record in value if record and not isinstance(record, float)]

# Do the same for pretrain_valid_data if needed
filtered_valid_data_list = {}
for key, value in valid_data_list.items():
    filtered_valid_data_list[key] = [record for record in value if record and not isinstance(record, float)]

filtered_train_data_list = [record for dataset in filtered_train_data_list.values() for record in dataset]
filtered_valid_data_list = [record for dataset in filtered_valid_data_list.values() for record in dataset]

#add back in context related data that will be validated on during finetuning
post_train_datasets = ['sciq', 'squad_v2', 'dolly15k']
post_train_data_list = []

#train finetune evaluation context records during pretraining
for dataset in post_train_datasets:
    pretrain_data = datasets_dict[dataset]['pretrain']
    valid_indices = valid_data_indices[dataset]

    for index in valid_indices:
        record = pretrain_data[index]
        if record and not isinstance(record, float):
            post_train_data_list.append(record)

#filter out records already captured during training and evaluation periods so I don't double dip
post_train_data_list = [record for record in post_train_data_list if record not in filtered_train_data_list]
post_train_data_list = [record for record in post_train_data_list if record not in filtered_valid_data_list]

combined_train = tokenizer.eos_token.join(filtered_train_data_list)
combined_valid = tokenizer.eos_token.join(filtered_valid_data_list)
combined_post_train = tokenizer.eos_token.join(post_train_data_list)

train_sequences = get_sequences(combined_train, tokenizer, seq_length=args['seq_length'])
valid_sequences = get_sequences(combined_valid, tokenizer, seq_length=args['seq_length'])
post_train_sequences = get_sequences(combined_post_train, tokenizer, seq_length=args['seq_length'])

train_epoch_steps = (len(train_sequences) / (args['batch_size'] * args['gradient_accumulation_steps']))
valid_epoch_steps = (len(valid_sequences) / (args['batch_size'] * args['gradient_accumulation_steps']))
post_train_epoch_steps  = (len(post_train_sequences) / (args['batch_size'] * args['gradient_accumulation_steps']))

max_train_steps = int(train_epoch_steps * args['epochs'])

train_dataset = datasets.Dataset.from_dict({"input_ids": train_sequences})
valid_dataset = datasets.Dataset.from_dict({"input_ids": valid_sequences})
post_train_dataset = datasets.Dataset.from_dict({"input_ids": post_train_sequences})

# Replace 'valid_dataset' with your current evaluation dataset variable
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
        learning_rate=args['learning_rate'],
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
        learning_rate=args['learning_rate'],
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

post_train_steps = max(1, int(np.round((initial_completed_steps/train_epoch_steps * post_train_epoch_steps),0)))

post_train_trainer = CustomTrainer(
    model=valid_trainer.model,
    max_eval_steps=post_train_steps,  # Pass the valid_steps here
    train_dataset=post_train_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size = args['batch_size'],
        per_device_eval_batch_size = args['batch_size'],
        gradient_accumulation_steps=args['gradient_accumulation_steps'],
        #warmup_steps=int(train_epoch_steps * args['warm_ratio']),
        #evaluation_strategy='steps',
        max_steps=post_train_steps,
        learning_rate=args['learning_rate'],
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

post_train_trainer.train()

with open("bits/bnb_config.json", "w") as f:
    json.dump(bnb_config.to_dict(), f)

post_train_trainer.model.save_pretrained('./bits')


