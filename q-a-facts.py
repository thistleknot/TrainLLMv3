import subprocess

from transformers import AutoTokenizer
from datasets import Dataset

import json
import os
import datasets
import torch
from tqdm import tqdm

# Replace with your actual API key
OPENAI_API_KEY="sk-111111111111111111111111111111111111111111111111"
OPENAI_API_BASE="http://192.168.3.122:5001/v1"

os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
os.environ['OPENAI_API_BASE']=OPENAI_API_BASE

# Initialize OpenAI API key
#openai.api_key = OPENAI_API_KEY

import openai

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
tokenizer.pad_token = tokenizer.eos_token

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
    tokenized_text = torch.tensor([token for sublist in dataset_dict["input_ids"] for token in (sublist + [eos_token_id])])[:-1]
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

def generate_qa_batch_with_chat_completion(contexts):
    generated_questions = []
    
    # Prepare the messages for ChatCompletion
    stringified_contexts = json.dumps(contexts)
    prompts = [
        {
            "role": "user",
            "content": stringified_contexts
        },
        {
            "role": "system",
            "content": "Complete every element of the array. Reply with an array of all completions."
        }
    ]
    
    # Perform the ChatCompletion
    stringified_batch_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompts,
        max_tokens=1000
    )
    
    # Extract the responses
    batch_completion = json.loads(stringified_batch_completion.choices[0].message.content)
    
    for completion in batch_completion:
        question = completion.strip().split("\n")[0]  # Adapt this line based on the structure of your output
        generated_questions.append(question)
    
    return generated_questions

# Your existing code to process in batches can stay the same, just replace the function

    
def generate_qa_batch_curl(contexts):
    try:
        generated_questions = []
        generated_answers = []
        
        for context in contexts:
            question_prompt = f"User:\n\n{context}\n\nInstruction: Provide a test question that can be answered using the above information.\n\nTest Question:\n"
            
            # Convert the prompt to a JSON object
            data = {
                "model": "gpt-3.5-turbo",
                "prompt": question_prompt,
                "max_tokens": 256,
                "temperature": 1.0
            }
            
            # Create the curl command
            curl_command = f"""curl http://192.168.3.122:5001/v1/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer {OPENAI_API_KEY}' -d {json.dumps(json.dumps(data))} --insecure"""
            
            print(f"Sending request with curl: {curl_command}")
            
            # Execute the curl command
            curl_response = subprocess.check_output(curl_command, shell=True).decode("utf-8")
            response_dict = json.loads(curl_response)
            generated_text = response_dict['choices'][0]['text'].strip().split("\n")[0]

            # Print the keys of the dictionary
            #print("Keys in the JSON response:")
            #for key in response_dict.keys():
            #    print(key)
            #print(f"Received response: {curl_response}")
            print(generated_text)
            
            # Parse the JSON response
            json_response = json.loads(curl_response)
            
            # Extract and store the question
            question = json_response["choices"][0]["text"].strip()
            generated_questions.append(question)
            
            # Generate answer (you can also use a curl command here in a similar way)
            generated_answers.append("Dummy answer")
        
        return generated_questions, generated_answers

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []  # Return empty lists to continue with the next batch


# Read data
input_file = "./source/"
selected_prompts = read_data(input_file)

# Create initial dataset
dataset = process_dataset(create_dataset(selected_prompts, tokenizer),tokenizer)

# Convert tokenized text to a list of strings for the API
tokenized_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in dataset['input_ids']]

# Initialize results_dict and tqdm progress bar
results_dict = {
    "contexts": [],
    "questions": []
}
pbar = tqdm(total=len(tokenized_texts))

# Process in batches (assuming batch size of 5 for demonstration)
batch_size = 1
for i in range(0, len(tokenized_texts), batch_size):
    try:
        batch_contexts = tokenized_texts[i:i+batch_size]
        
        # Generate questions and answers
        #generated_questions, generated_answers = generate_qa_batch_with_chat_completion(batch_contexts)
        generated_questions, generated_answers = generate_qa_batch_curl(batch_contexts)
        
        # Store in results_dict
        results_dict["contexts"].extend(batch_contexts)
        results_dict["questions"].extend(generated_questions)
        #results_dict["answers"].extend(generated_answers)
        
        # Update tqdm progress bar
        pbar.update(len(batch_contexts))

    except Exception as e:
        print(f"An error occurred in the batch process: {e}")

# Close tqdm progress bar
pbar.close()


# Save to JSON file
with open("results.json", "w") as f:
    json.dump(results_dict, f, indent=4)

print("Results saved to results.json")
