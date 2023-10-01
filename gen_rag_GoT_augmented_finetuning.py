import subprocess
import time
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

from sklearn.neighbors import NearestNeighbors, KernelDensity
import numpy as np

server_ip = "192.168.3.17"
user = "root"
script_path = "source /home/user/miniconda3/etc/profile.d/conda.sh && cd /home/user/text-generation-webui/ && /home/user/miniconda3/envs/textgen/bin/python /home/user/text-generation-webui/server.py"

HOST = '192.168.3.17:5000'
URI = f'http://{HOST}/api/v1/generate'

n_samples = 1
top_k = 10  # The number of neighbors you want to consider.

# Load dataset and model
dataset = load_dataset("Abirate/english_quotes")
quotes = [item['quote'] for item in dataset['train']]

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
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(quotes, show_progress_bar=True)
    save_embeddings_to_db(embeddings)
else:
    # Load embeddings if they exist
    embeddings = load_embeddings_from_db()

# Fit Kernel Density model
knn = NearestNeighbors(n_neighbors=top_k + 1, metric='cosine')  # +1 to include the sample itself
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(embeddings)
knn.fit(embeddings)
# Sample from dense regions
samples = kde.sample(n_samples=n_samples)

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

    selected_quote = quotes[indices[0]]  # The quote corresponding to the sample
    final_samples_dict[selected_quote] = [(quotes[idx], 1 - filtered_distances[filtered_indices.index(idx)]) for idx in selected_indices]

promptsArray = []

#works with qwen, speechless, athena, and synthia
def format_universal_prompt(context: str) -> str:
    """
    Formats and returns a Universal Analysis prompt string.
    
    :param context: The context string to be included in the prompt.
    :return: A formatted prompt string.
    """
    return f"""
Context:

{context}
Instruction:

When responding, follow the processes outlined below.
Succinct points.
Use 3rd person objective.
Solely source from what is being expressed within the context.

Universals (thesis)
Identify no more than three core ideas, themes, or meanings shared (expressed) across any subset of sentences (i.e. universals: Aristotle; forms: Plato; Archetypes: Jung).

Constraints (elenchus, anti-thesis)
Identify and explain up to two contrary ideas expressed within any single sentence that differentiates itself from a shared meaning.

Conclusion (Synthesis)
Integrate the derived information to form premises (syllogistic and causal reasoning) to construct a clear, cohesive generalized conclusion.

Response:

Universals
"""

#works with athena
def format_GoT_prompt(context):
    return f"""
Context:

{context}
Instruction:

When responding, follow the processes outlined below.
Be succinct.
Use 3rd person objective.
Solely source from what is being expressed within the context.

Graph of Thoughts (GoT) process
 Nodes: Identify 2 to 3 main ideas (e.g. nouns, verbs, adjectives) expressed across multiple sentences within the context.
 Vertices: Limit/constrain/shape these ideas by identifying 1 to 2 contraries to act as axioms (edges) of understanding (e.g. hot/cold across a discussion about metals) that differentiate across these ideas.
Synthesis process
 Premises: Describe the product of these interactions between nodes (vertices).
 Rank: Identify the premises with most/least weight (using scientific consensus, else popular opinion).
 Generalize: Synthesize the interrelated information.

Response:
"""

def format_combined_prompt(context: str) -> str:
    return f"""
Context:

{context}
Instruction:

When responding, follow the process as outlined below.
Be succinct.
Use 3rd person objective.
Solely source from the context.
Syllogistic reasoning: Deduce conclusions from premises; each premise is a statement claimed to be true.
Inductive reasoning: Evaluate the likelihood of a premise being true by the sum liklihood of all it's conditional interaction weights.
Causal reasoning: Identify directional interacting relationships between between nodes (across edges).
Node: Represents an idea.
Edge: Represents the relationship and direction between two ideas.
Interaction: Represents edge weights, or merit of these ideas.
Conditions:  Determining factors presumed of context.

Graph of Thoughts (GoT) process

Identify Core Ideas (Nodes/Thesis)
 - Identify and describe 2 to 3 main ideas, themes, or shared meanings expressed across at least 2 sentences within the context (e.g. universals: Aristotle; forms: Plato; Archetypes: Jung; e.g. nouns, verbs, adjectives, i.e. a platonic class/form of being, i.e. Archetypes: when some attribute is True across multiple instances of some percieved class).

Identify Constraints (Vertices, edges, Anti-thesis)
 - Constraints: Identify and explain up to two contrasting ideas--within any single sentence--that constrain, limit, and/or differentiate the universal (e.g. metal and hot/cold, i.e. ways a form can vary, differentiating characteristics) which will serve as axioms of understanding that differentiate and connect across these ideas.
 - Conditions: Identify determining factors between interacting ideas for the idea to be active. 
 - Activation Function: Use scientific consensus (else popular opinion) to derive probabilities (likelihoods) of each condition (constraints) being True, of which all conditions percents should add up to 1.  A sum product of greater than 50% is interpreted as a universal having met all it's conditions to be True for a specific directional interaction (i.e. a specific moment in time where two nodes directly interacted with one another).

Synthesis
 - Premises: Identify and describe associated interactions between universals by exploring and evaluating node edge direction and weights.
 - Rank: Identify the premises with the most and least weight by interpreting the conditional interaction weights (i.e. sum product between two nodes edges). 
 - Conclusion: Apply deductive logic to integrate and synthesize the premises and derive conclusions from them, while utilizing inductive reasoning to assess their merits, evaluating the probability of the necessary conditions, to formulate a coherent and generalized syllogism.
 
Response:
"""

promptsArray = []
for sample, selected_matches in final_samples_dict.items():
    context = ""
    for match, weight in selected_matches:
        context += f"{match}\n"
    
    promptsArray.append(format_combined_prompt(context))

promptsarray = [p for p in promptsArray if len(p) <= 1024]

#print(promptsArray[1])

# For local streaming, the websockets are hosted without ssl - http://
#HOST = '192.168.3.18:5000'

def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 4096,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
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

len([q for q in quotes if len(q) <128])

# Initialize tqdm with the total number of batches
#total_batches = len(promptsArray) // bs + int(len(promptsArray) % bs != 0)

# Define the commands to be executed remotely
kill_command = f"ssh {user}@{server_ip} -C 'pkill -f \"server.py\" || kill -9 $(ps aux | grep \"server.py\" | grep -v grep | awk '{{print $2}}')'"
print(kill_command)

commands = [
    ["deci",f"{script_path} --api --listen --xformers --sdp-attention --trust-remote-code --disk-cache-dir /data/tmp --use_double_quant --quant_type nf4 --numa --load-in-4bit --settings settings-template.yaml --model /home/user/text-generation-webui/models/DeciLM-6b-instruct/"],
    ["intern",f"{script_path} --api --listen --xformers --sdp-attention --trust-remote-code --disk-cache-dir /data/tmp --use_double_quant --quant_type nf4 --numa --load-in-4bit --settings settings-template.yaml --model /home/user/text-generation-webui/models/internlm-chat-7b/"],
    ["synthia",f"{script_path} --api --listen --extensions openai --use_fast --xformers --sdp-attention --n-gpu-layers 128 --threads 8 --cpu --n_ctx 4096 --numa --model /home/user/text-generation-webui/models/synthia-7b-v1.3.Q4_K_M.gguf"],
    ["athena",f"{script_path} --api --listen --extensions openai --use_fast --xformers --sdp-attention --n-gpu-layers 128 --threads 8 --cpu --n_ctx 4096 --numa --model /home/user/text-generation-webui/models/Athena-v3.q5_K_M.gguf"],
    ["speechless",f"{script_path} --api --listen --extensions openai --use_fast --xformers --sdp-attention --n-gpu-layers 128 --threads 8 --cpu --n_ctx 4096 --numa --model /home/user/text-generation-webui/models/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf"],
    ["qwen",f"{script_path} --api --listen --xformers --sdp-attention --trust-remote-code --disk-cache-dir /data/tmp --use_double_quant --quant_type nf4 --numa --load-in-4bit --settings settings-template.yaml --model /home/user/text-generation-webui/models/Qwen-LLaMAfied-7B-Chat/"]
]

for name_, command in commands:
    print(name_)
    full_command = f"ssh {user}@{server_ip} -C '{command}'"
    print(full_command)
    process = subprocess.Popen(full_command, shell=True)
    time.sleep(20)  # Let the process start

    # Sleep for 3 seconds
    responses = []
    with tqdm(total=len(promptsArray), desc='Processing prompts') as progress_bar:
        for p in promptsArray:
            prompt, result = run(p)
            print(prompt + result)
            responses.append([prompt, result])
            progress_bar.update(1)

    # Convert the responses to a DataFrame
    df_responses = pd.DataFrame({
        "Prompt": [response[0] for response in responses],  # response[0] will give you the prompt
        "Response": [response[1].strip() for response in responses]  # response[1] will give you the result
    })

    # Save the DataFrame to a CSV file
    df_responses.to_json(f"{name_}_responses.json", orient='records', lines=False, indent=4)
    subprocess.run(kill_command, shell=True)
