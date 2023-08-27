from functions import process_dataset
from common_imports import tokenizer  # Assuming you've defined or imported the tokenizer somewhere
import openai
import json
import os

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import openai
import json
import os

# Function to read data
def read_data(input_file):
    selected_prompts = []
    for file_name in os.listdir(input_file):
        file_path = os.path.join(input_file, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                selected_prompts.append(content)
    return selected_prompts

# Function to create dataset (assuming tokenizer is defined)
def create_dataset(selected_prompts, tokenizer):
    input_ids_list = []
    for text in selected_prompts:
        tokenized_text = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').input_ids[0].tolist()
        input_ids_list.append(tokenized_text)

    dataset_dict = {
        "input_ids": input_ids_list,
    }
    
    return Dataset.from_dict(dataset_dict)

# Function to batch summarize tokenized text
def batch_summarize_tokenized(tokenized_texts):
    stringified_texts = json.dumps(tokenized_texts)
    
    prompts = [
        {
            "role": "user",
            "content": stringified_texts
        }
    ]
    
    batch_instruction = {
        "role": "system",
        "content": "Summarize every element of the array. Reply with an array of all summaries."
    }
    
    prompts.append(batch_instruction)
    
    response = openai.ChatCompletion.create(
        model="flan-t5-xl",
        messages=prompts,
        max_tokens=1000
    )
    
    stringified_batch_summaries = response.choices[0].message.content
    batch_summaries = json.loads(stringified_batch_summaries)
    
    return batch_summaries

# Main Code

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Initialize OpenAI API key
openai.api_key = "OPENAI_API_KEY"  # Replace with your actual API key

# Read data
input_file = "your_input_directory_here"  # Replace with your actual directory
selected_prompts = read_data(input_file)

# Create initial dataset
dataset = create_dataset(selected_prompts, tokenizer)

# Process dataset to get tokenized and chunked text
# (Add your other parameters as needed)
train_dataset, valid_dataset, eval_subset = process_dataset(dataset, tokenizer)

# Convert tokenized text to a list of strings for the API
tokenized_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in train_dataset['input_ids']]

# Batch summarize
batch_summaries = batch_summarize_tokenized(tokenized_texts)

# Output the generated summaries
print("Generated Summaries: ", batch_summaries)

