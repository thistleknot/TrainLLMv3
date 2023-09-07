import subprocess
import ctranslate2
import transformers
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, Trainer, EvalPrediction, TrainingArguments, TrainerControl, TrainerState, TrainerCallback, logging, pipeline, DataCollatorForLanguageModeling

import json
import os
import datasets
import torch
from tqdm import tqdm

device_map = 'cuda'

tokenizer = transformers.AutoTokenizer.from_pretrained("/data/text-generation-webui/models/t5-large-generation-squad-QuestionAnswer/")
#tokenizer = transformers.AutoTokenizer.from_pretrained("/data/text-generation-webui/models/t5-large-generation-race-QuestionAnswer/")
translator = ctranslate2.Translator("/data/text-generation-webui/models/t5-large-generation-squad-QuestionAnswer_ct/",device=device_map)
#translator = ctranslate2.Translator("/data/text-generation-webui/models/t5-large-generation-race-QuestionAnswer_ct/",device=device_map)

# Function to read data
def read_data(input_file):
    selected_prompts = []
    for file_name in os.listdir(input_file):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_file, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    selected_prompts.append(content)
    return selected_prompts

def process_dataset(dataset_dict, tokenizer, STRIDE_LENGTH=128, BLOCK_SIZE=256):
    
    eos_token_id = tokenizer.eos_token_id
    tokenized_text = torch.tensor([token for sublist in dataset_dict["input_ids"] for token in (sublist)])
    total_length = len(tokenized_text)

    # 2. Generate stride lengths dynamically
    stride_lengths = [STRIDE_LENGTH]
    while stride_lengths[-1] > 1:
        stride_lengths.append(stride_lengths[-1] // 2)

    # 3. Choose the stride length with the highest modulus when divided by total_length
    STRIDE_LENGTH = max(stride_lengths, key=lambda x: total_length % x)
    print("Optimally derived STRIDE_LENGTH", STRIDE_LENGTH)

    # 3. Split the tokenized text into sequences of length `block_size`
    input_ids_list = []

    for i in range(0, len(tokenized_text), STRIDE_LENGTH):
        end = i + BLOCK_SIZE
        partial_sequence = tokenized_text[i:min(end, len(tokenized_text))]

        # If we're at the last chunk and it's shorter than BLOCK_SIZE
        if end >= len(tokenized_text):
            num_padding = BLOCK_SIZE - len(partial_sequence)
            padding = [eos_token_id] * num_padding
            partial_sequence = torch.cat([partial_sequence, torch.tensor(padding, dtype=torch.long)], dim=0)

        input_ids_list.append(partial_sequence)

    # 4. Create attention masks
    attention_mask_list = [[1] * BLOCK_SIZE for _ in input_ids_list]

    # 5. Use the same tokenized sequences for labels
    labels_list = input_ids_list.copy()
    input_ids = [seq.tolist() for seq in input_ids_list]
    attention_mask = attention_mask_list
    labels = [seq.tolist() for seq in labels_list]
    
    print('total_length', total_length)
    
    dataset = datasets.Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    })
    
    return dataset

# Function to create dataset (assuming tokenizer is defined)
def create_dataset(selected_prompts, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for text in selected_prompts:
        tokenized_text = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').squeeze()
        attention_mask = [1] * len(tokenized_text) # Adjusting for variable lengths
        input_ids_list.append(tokenized_text.tolist())  # Truncate or pad as needed
        
        attention_mask_list.append(attention_mask)
        labels_list.append(tokenized_text.tolist())  # Using the same tokenized sequence for labels

    dataset_dict = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }
    
    return dataset_dict

# Read data
input_file = "./source/"
selected_prompts = read_data(input_file)

# Create initial dataset
dataset = process_dataset(create_dataset(selected_prompts, tokenizer),tokenizer)

# Convert tokenized text to a list of strings for the API
contexts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in dataset['input_ids']]
print(len(contexts))

num_q = 1
if(False):
    num_c = 10
else:
    num_c = len(contexts)

bs = 16

device_map = 'cuda'

input_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(c)) for c in contexts]

input_tokens = [i for i in input_tokens if len(i) <= 512]

input_tokens = input_tokens[0:num_c]

# Initialize an empty list to hold the converted contexts
converted_contexts = []

# Iterate over each token list in input_tokens
for tokens in input_tokens:
    # Convert tokens to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Decode the IDs back into text form
    context_text = tokenizer.decode(input_ids)
    
    # Append the context text to the list of converted contexts
    converted_contexts.append(context_text)

final_output = {}

ctranslate2.set_random_seed(11)

#results = translator.translate_batch(input_tokens, max_batch_size=bs, asynchronous=True, batch_type="tokens", num_hypotheses=num_q)
# Initialize an empty list to store all results
all_results = []

# Split the input_tokens into sub-batches of size 2
for i in range(0, len(input_tokens), bs):
    sub_batch = input_tokens[i:i+bs]
    
    # Translate the sub-batch
    results = translator.translate_batch(
        sub_batch, 
        max_batch_size=bs, 
        asynchronous=True, 
        batch_type="examples", 
        num_hypotheses=num_q
    )
    
    # Append the results to all_results
    all_results.extend(results)

# Iterate through each context and its corresponding translation result
for idx, (context_, async_result) in enumerate(zip(converted_contexts, results)):
    # Initialize an empty list to hold QA pairs for the current context
    qa_pairs = []
    
    # Get the actual result from the AsyncTranslationResult
    for hypothesis in async_result.result():
        # Decode the tokens back to text
        output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(hypothesis['tokens']))
        
        # Split the output text into question and answer
        #question, answer = output_text.split('<sep>')
        question_answer = output_text#.split('<sep>')
        
        # Create a QA pair
        qa_pair = {
            'question_answer': question_answer,
            #'answer': answer.strip()
        }
        
        # Append the QA pair to the list of QA pairs for the current context
        qa_pairs.append(qa_pair)

    # Store the context and its QA pairs in the final_output dictionary
    # Use the index as the key and include the actual context text as part of the value
    final_output[f"{idx}"] = {
        'context': context_,
        'qa_pairs': qa_pairs
    }

# Save the final_output dictionary to an indented JSON file
with open('final_output.json', 'w') as f:
    json.dump(final_output, f, indent=4)

print(final_output)

