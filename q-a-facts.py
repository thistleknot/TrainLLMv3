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

#from vars

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

# Function to generate questions and answers using API with batch processing
def generate_qa_batch(contexts):
    generated_questions = []
    generated_answers = []
    
    question_messages = []
    for context in contexts:
        question_prompt = f"""User:

{context}

Instruction: Provide a test question that can be answered using the above information.

Test Question:\n"""
        question_messages.append({"role": "user", "content": question_prompt})
    
    question_messages.append({"role": "system", "content": "Complete every element of the array. Reply with an array of all completions."})
    
    question_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=question_messages,
        max_tokens=512 * len(contexts)
    )
    
    question_batch_completion = json.loads(question_response.choices[0].message.content)
    generated_questions.extend(question_batch_completion)
    
    answer_messages = []
    for i, context in enumerate(contexts):
        answer_prompt = f"""User:

{context}

Question: {generated_questions[i]}

Answer:\n"""
        answer_messages.append({"role": "user", "content": answer_prompt})
    
    answer_messages.append({"role": "system", "content": "Complete every element of the array. Reply with an array of all completions."})
    
    answer_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=answer_messages,
        max_tokens=512 * len(contexts)
    )
    
    answer_batch_completion = json.loads(answer_response.choices[0].message.content)
    generated_answers.extend(answer_batch_completion)
    
    return generated_questions, generated_answers

# Function to synthesize facts using API with batch processing
def synthesize_facts(qa_pairs):
    synthesized_facts = []
    
    fact_messages = []
    for q, a in qa_pairs:
        fact_prompt = f"""User:

I have a question and an answer:
Question: {q}
Answer: {a}

Instruction: Synthesize this information into a factual statement.

Fact:\n"""
        fact_messages.append({"role": "user", "content": fact_prompt})
    
    fact_messages.append({"role": "system", "content": "Complete every element of the array. Reply with an array of all completions."})
    
    fact_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=fact_messages,
        max_tokens=256 * len(qa_pairs)
    )
    
    fact_batch_completion = json.loads(fact_response.choices[0].message.content)
    synthesized_facts.extend(fact_batch_completion)
    
    return synthesized_facts

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
tokenizer.pad_token = tokenizer.eos_token

# Read data
input_file = "./source/"  # Replace with your actual directory
selected_prompts = read_data(input_file)

# Create initial dataset
dataset = create_dataset(selected_prompts, tokenizer)
processed_dataset = process_dataset(dataset, tokenizer)

# Convert tokenized text to a list of strings for the API
tokenized_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in processed_dataset['input_ids']]

# Initialize results dictionary
results_dict = {
    "contexts": [],
    "questions": [],
    "answers": [],
    "facts": []
}

# Initialize tqdm progress bar
pbar = tqdm(total=len(tokenized_texts))

# Process in batches (assuming batch size of 5 for demonstration)
batch_size = 5
for i in range(0, len(tokenized_texts), batch_size):
    batch_contexts = tokenized_texts[i:i+batch_size]
    
    # Generate questions and answers
    generated_questions, generated_answers = generate_qa_batch(batch_contexts)
    
    # Synthesize facts
    qa_pairs = list(zip(generated_questions, generated_answers))
    synthesized_facts = synthesize_facts(qa_pairs)
    
    # Store in results_dict
    results_dict["contexts"].extend(batch_contexts)
    results_dict["questions"].extend(generated_questions)
    results_dict["answers"].extend(generated_answers)
    results_dict["facts"].extend(synthesized_facts)
    
    # Update tqdm progress bar
    pbar.update(len(batch_contexts))

    # Display input to output for each batch
    print("\nBatch Results:")
    for j in range(len(batch_contexts)):
        print(f"Context: {batch_contexts[j]}")
        print(f"Question: {generated_questions[j]}")
        print(f"Answer: {generated_answers[j]}")
        print(f"Fact: {synthesized_facts[j]}")
        print("----")

# Close tqdm progress bar
pbar.close()

# Save to JSON file
with open("results.json", "w") as f:
    json.dump(results_dict, f, indent=4)

print("Results saved to results.json")