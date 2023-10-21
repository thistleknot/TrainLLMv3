import subprocess
import transformers
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, Trainer, EvalPrediction, TrainingArguments, TrainerControl, TrainerState, TrainerCallback, logging, pipeline, DataCollatorForLanguageModeling
import os
import sqlite3
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import json
import requests
from sklearn.neighbors import NearestNeighbors, KernelDensity
import torch

print(torch.cuda.is_available())

from tqdm import tqdm

n_samples = 100
top_k = 10  # The number of neighbors you want to consider.

"""
In your code, you are using the Kernel Density Estimation (KDE) model to sample data. The number of samples you generate depends on the n_samples variable, which you've set to 1500. However, the actual number of valid samples might be lower if the KDE model is unable to generate samples in certain regions of the feature space with a cosine similarity above 0.

Filtering Based on Cosine Similarity:
You are filtering the indices and distances based on cosine similarity. If there are no neighbors with cosine similarity greater than 0 for a given sample, that sample is skipped. This filtering might reduce the number of valid samples.

Weighted Random Selection:
You are performing weighted random selection of neighbors based on cosine similarity. If the weights are extremely low for some samples, they might not get selected, further reducing the number of valid samples.

Minimum Number of Neighbors:
You have set num_to_select to a minimum of 2 neighbors, but if a sample doesn't have at least 2 neighbors with cosine similarity greater than 0, it might result in fewer samples.
"""

# Load dataset and model
#dataset = load_dataset("Abirate/english_quotes")
#quotes = [item['quote'] for item in dataset['train']]

# Check if SQLite database exists, if not, generate and save embeddings
db_path = "chunks.sqlite"

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

def process_dataset(dataset_dict, tokenizer, STRIDE_LENGTH=96, BLOCK_SIZE=128):

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

    dataset = Dataset.from_dict({
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
        tokenized_text = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').squeeze()
        attention_mask = [1] * len(tokenized_text) # Adjusting for variable lengths
        input_ids_list.append(tokenized_text.tolist())  # Truncate or pad as needed

        attention_mask_list.append(attention_mask)
        labels_list.append(tokenized_text.tolist())  # Using the same tokenized sequence for labels

    dataset_dict = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "text": text
    }

    return dataset_dict

def save_embeddings_to_db(embeddings, db_path="chunks.sqlite"):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, embedding BLOB)''')
        for emb in embeddings:
            emb_bytes = emb.tobytes()
            cursor.execute("INSERT INTO embeddings (embedding) VALUES (?)", (emb_bytes,))
        conn.commit()

def load_embeddings_from_db(db_path="chunks.sqlite"):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM embeddings")
        rows = cursor.fetchall()
    return [np.frombuffer(row[1], dtype=np.float32) for row in rows]

input_file = "/home/user/input/"
selected_prompts = read_data(input_file)

tokenizer = transformers.AutoTokenizer.from_pretrained('/home/user/text-generation-webui/models/Llama-2-7b-hf/')

dataset_ = create_dataset(selected_prompts, tokenizer)
dataset = process_dataset(dataset_,tokenizer)

chunks = [tokenizer.decode(seq, skip_special_tokens=True) for seq in dataset['input_ids']]

print(len(chunks))
print(type(chunks))

if not os.path.exists(db_path):
    # Generate embeddings if they don't exist
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Read data

    embeddings = sentence_model.encode(chunks, show_progress_bar=True)
    save_embeddings_to_db(embeddings)
else:
    # Load embeddings if they exist
    embeddings = load_embeddings_from_db()

# Fit Kernel Density model
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(embeddings)
knn = NearestNeighbors(n_neighbors=top_k + 1, metric='cosine')  # +1 to include the sample itself
knn.fit(embeddings)
# Sample from dense regions
samples = kde.sample(n_samples=n_samples)

print(len(samples))

final_samples_dict = {}

for i, sample in enumerate(samples):
    distances, indices = knn.kneighbors([sample])
    indices = indices[0]
    distances = distances[0]
    
    # Filtering indices based on cosine similarity greater than 0
    filtered_indices = [idx for idx, dist in zip(indices, distances) if 1 - dist > 0]
    filtered_distances = [dist for dist in distances if 1 - dist > 0]
    
    if len(filtered_indices) == 0:
        continue  # Skip if there are no neighbors with cosine similarity > 0
    
    # Compute weights based on cosine similarity
    cosine_similarities = [1 - dist for dist in filtered_distances]
    weights = np.array(cosine_similarities)
    weights /= weights.sum()  # Normalize the weights
    
    # Perform weighted random selection of neighbors
    num_to_select = min(3, len(filtered_indices))
    selected_indices = np.random.choice(filtered_indices, size=num_to_select, replace=False, p=weights)
    
    selected_quote = chunks[indices[0]]  # The quote corresponding to the sample
    final_samples_dict[selected_quote] = [(chunks[idx], 1 - filtered_distances[filtered_indices.index(idx)]) for idx in selected_indices]

def apply_context(context):
    with open("prompt_template.txt", "r") as file:
        template = file.read()

    # Identify the placeholder in your template. 
    # This could be a unique sequence of characters that doesn't appear anywhere else in your template.
    placeholder = '{context}'

    # Replace the placeholder with the actual context. 
    # This is a direct string replacement, avoiding the issue with `str.format()` entirely.
    customized_template = template.replace(placeholder, context)

    #return template.format(context=context)
    return customized_template

promptsArray = []

for sample, selected_matches in final_samples_dict.items():
    context = ""
    for match, weight in selected_matches:
        context += f"{match}\n"
        prompt = apply_context(context)

    #prompt = f"Context:\n\n{context}\n\nInstruction:\n\nWhen responding, follow the processes outlined below.\nBe succinct.\nUse 3rd person objective.\nSolely source from what is being expressed within the context.\n\nGraph of Thoughts (GoT) process\n Nodes: Identify 2 to 3 common ideas (e.g. nouns, verbs, adjectives).\n Vertices: Identify 1 to 2 contraries as axioms (edges) of understanding (e.g. ranges such as hot/cold) across these ideas.\nSynthesis process\n Premises: Use nodes and vertices to form premises.\n Rank: Identify the premises with most/least weight (using scientific consensus, else popular opinion).\n Generalize: Synthesize the interrelated information.\n\nResponse:\n"
    #prompt = f"Context:\n\n{context}\nInstruction:\n\nSuccinct points.\n3rd person objective.\n\nUniversals (thesis)\nIdentify no more than three core ideas, themes, or meanings shared (expressed) across any subset of sentences (i.e. universals: Aristotle; forms: Plato; Archetypes: Jung).\n\nConstraints (elenchus, anti-thesis)\nIdentify and explain up to two contrary ideas expressed within any single sentence that differentiates itself from a shared meaning.\n\nConclusion (Synthesis)\nIntegrate the derived information to form premises (syllogistic and causal reasoning) to construct a clear, cohesive generalized conclusion.\n\nResponse:\n\nUniversals\n\n"
    # Prepare the prompt and append it to the promptsArray
    #prepped_prompt = f"Context:\n\n{all_quotes_str}\nInstruction:\n\nLet's think step by step.\nStep 1. Identify the facts expressed.\nStep 2. Identify the common themes amongst the facts.\nStep 3. Generalize the information.\n\nResponse:\n\n"
    #prompt = f"Context:\n\n{all_quotes_str}\nInstruction:\n\nSource information from sentences within the context\nProvide clear and concise explanations.\n3rd person objective.\nProvide an outline containing the following\n\nCommon Factors\nIdentify no more than three core ideas, themes, or meanings shared/expressed across any subset of sentences (i.e. universals: Aristotle; forms: Plato; Archetypes: Jung).\nCounter Examples\nIdentify and explain up to two contrary ideas expressed within any single sentence that differentiates itself from a shared meaning.\nConclusion\nIntegrate the derived information to form premises (syllogistic and causal reasoning) in support of a generalized conclusion.\n\nResponse:\n\nCommon Factors\n"
    #prompt = f"Context:\n\n{all_quotes_str}\n\nInstruction:\n\nIdentify and unpack the common archetypal (Jungian) meaning(s) via synthesis from the above quotes.\n\nResponse:\n\n"
    promptsArray.append(prompt)
    
#promptsarray = [p for p in promptsArray if len(p) <= 6144]
promptsarray = promptsArray

print(len(promptsarray))

# For local streaming, the websockets are hosted without ssl - http://
#HOST = '192.168.3.17:5000'
HOST = '127.0.0.1:5000'
URI = f'http://{HOST}/api/v1/generate'

def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 3072,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.95,
        'typical_p': .9,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 50,
        'min_length': 0,
        'no_repeat_ngram_size': 3,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': True,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 4096,
        'ban_eos_token': False,
        'custom_token_bans': '',
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)
    
    result = response.json()['results'][0]['text']
    return([prompt,result])


# Initialize tqdm with the total number of batches
#total_batches = len(promptsArray) // bs + int(len(promptsArray) % bs != 0)

progress_bar = tqdm(total=len(promptsArray), desc='Processing prompts')

responses = []
for p in promptsArray:
    #print(p)
    prompt, result = run(p)
    print(prompt + result)
    responses.append([prompt,result])
    # Update tqdm progress bar
    progress_bar.update(1)
# Close tqdm progress bar
progress_bar.close()

# Convert the responses to a DataFrame
df_responses = pd.DataFrame({
    "Prompt": [response[0] for response in responses],  # response[0] will give you the prompt
    "Response": [response[1].strip() for response in responses]  # response[1] will give you the result
})

# Save the DataFrame to a CSV file
df_responses.to_json("responses.json", orient='records', lines=False, indent=4)

