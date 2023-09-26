import openai
import os
import sqlite3
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
from gptcache import cache
#from gptcache.adapter import openai
import json
import requests

from tqdm import tqdm

n_samples = 500
top_k = 10  # The number of neighbors you want to consider.

openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # can be anything
openai.api_base = "http://127.0.0.1:8000/v1"

cache.init()
cache.set_openai_key()

# Load dataset and model
dataset = load_dataset("Abirate/english_quotes")
quotes = [item['quote'] for item in dataset['train']]
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if SQLite database exists, if not, generate and save embeddings
db_path = "embeddings.sqlite"

np.mean([len(q) for q in quotes])

def save_embeddings_to_db(embeddings, db_path="embeddings.sqlite"):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, embedding BLOB)''')
        for emb in embeddings:
            emb_bytes = emb.tobytes()
            cursor.execute("INSERT INTO embeddings (embedding) VALUES (?)", (emb_bytes,))
        conn.commit()

def load_embeddings_from_db(db_path="embeddings.sqlite"):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM embeddings")
        rows = cursor.fetchall()
    return [np.frombuffer(row[1], dtype=np.float32) for row in rows]

if not os.path.exists(db_path):
    # Generate embeddings if they don't exist
    embeddings = sentence_model.encode(quotes, show_progress_bar=True)
    save_embeddings_to_db(embeddings)
else:
    # Load embeddings if they exist
    embeddings = load_embeddings_from_db()

from sklearn.neighbors import NearestNeighbors, KernelDensity
import numpy as np

# Fit Kernel Density model
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(embeddings)
# Sample from dense regions
samples = kde.sample(n_samples=n_samples)

final_samples_dict = {}

for i, sample in enumerate(samples):
    knn = NearestNeighbors(n_neighbors=top_k + 1, metric='cosine')  # +1 to include the sample itself
    knn.fit(embeddings)
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
    
    selected_quote = quotes[indices[0]]  # The quote corresponding to the sample
    final_samples_dict[selected_quote] = [(quotes[idx], 1 - filtered_distances[filtered_indices.index(idx)]) for idx in selected_indices]

promptsArray = []

for sample, selected_matches in final_samples_dict.items():
    all_quotes_str = ""
    for match, weight in selected_matches:
        all_quotes_str += f"{match}\n"
    # Prepare the prompt and append it to the promptsArray
    prompt = f"Context:\n\n{all_quotes_str}\n\nInstruction:\n\nIdentify and unpack the common archetypal (Jungian) meaning(s) via synthesis from the above quotes.\n\nResponse:\n\n"
    promptsArray.append(prompt)
    
promptsarray = [p for p in promptsArray if len(p) <= 640]

cod_prompt = f"""
Summary Guidelines
You will generate increasingly concise, entity-dense summaries of the response. Repeat the following 2 steps 5 times.

Step 1
Identify 1-3 informative Entities (delimited by ";") from the response which are missing from the previously generated summary.

Step 2
Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

Definition of a Missing Entity
Relevant: to the main story.
Specific: descriptive yet concise (5 words or fewer).
Novel: not in the previous summary.
Faithful: present in the response.
Anywhere: located anywhere in the response.
Guidelines
The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
Make space with fusion, compression, and removal of uninformative phrases like "the response discusses."
The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the response.
Missing entities can appear anywhere in the new summary.
Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Answer in JSON
The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
"""

#print(promptsArray[1])

# For local streaming, the websockets are hosted without ssl - http://
HOST = '192.168.3.18:5000'
URI = f'http://{HOST}/api/v1/generate'

def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 2048,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'custom_token_bans': '',
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)
    
    result = response.json()['results'][0]['text']
    return([prompt,result])


len([q for q in quotes if len(q) <128])

# Initialize tqdm with the total number of batches
#total_batches = len(promptsArray) // bs + int(len(promptsArray) % bs != 0)

progress_bar = tqdm(total=len(promptsArray), desc='Processing prompts')

responses = []
for p in promptsArray:
    #print(p)
    prompt, result = run(p)
    print(prompt + result)
    responses.append([prompt, result])

    """
    response = openai.Completion.create(
        model="text-davinci-003", # currently can be anything
        prompt=p,
        max_tokens=4096,
        temperature=0.7
    )
    responses.append(response)
    """
    # Update tqdm progress bar
    progress_bar.update(1)
# Close tqdm progress bar
progress_bar.close()

#print(f"Number of prompts: {len(promptsArray)}")
#print(f"Number of choices in response: {len(responses)}")

#for response in responses:
#    print(f"Prompt: {response['prompt']}")
#    print(f"Response: {response['result']}")
    
# Convert the responses to a DataFrame
df_responses = pd.DataFrame({
    "Prompt": [response[0] for response in responses],  # response[0] will give you the prompt
    "Response": [response[1].strip() for response in responses]  # response[1] will give you the result
})

# Save the DataFrame to a CSV file
df_responses.to_json("responses.json", orient='records', lines=False, indent=4)

