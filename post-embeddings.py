import os
import json
import faiss
import numpy as np
from tqdm import tqdm

# Function to load JSONB file
def load_jsonb(file_path):
    with open(file_path, "rb") as f:
        return json.loads(f.read().decode('utf-8'))

# Function for embedding text (you can replace this with your actual embedding function)
def embed_text(text):
    return {'embedding': np.random.rand(768)}  # Replace with your actual embedding logic

# Load the dataset
dataset = load_jsonb("dataset.jsonb")

# Load the saved embeddings
with open("dataset.jsonb", "rb") as f:
    final_data = json.load(f)

# Get the question and context embeddings
question_embeddings = [np.array(emb).astype('float32') for emb in final_data['question_embedding']]
context_embeddings = [np.array(emb).astype('float32') for emb in final_data['context_embedding']]

print("Loaded {} question embeddings".format(len(question_embeddings)))
question_embeddings = [np.array(emb) for emb in final_data['question_embedding']]
print("Loaded {} context embeddings".format(len(context_embeddings)))

# Get the contexts and questions
contexts = dataset['contexts']
questions = dataset['questions']

# File path for the FAISS index
faiss_index_file_path = 'my_index.faiss'

# Check if FAISS index file already exists
if os.path.exists(faiss_index_file_path):
    print("Loading existing FAISS index.")
    index = faiss.read_index(faiss_index_file_path)
else:
    print("Generating new FAISS index.")
    # Commenting out re-generation of embeddings
    # question_embeddings = np.array([embed_text(text)['embedding'] for text in tqdm(questions, desc="Embedding questions")]).astype('float32')
    # context_embeddings = np.array([embed_text(text)['embedding'] for text in tqdm(contexts, desc="Embedding contexts")]).astype('float32')
    
    # Initialize the FAISS index
    dimension = context_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)

    # Convert list of arrays to a 2D array
    context_embeddings_matrix = np.vstack(context_embeddings)

    # Add vectors to index
    index.add(context_embeddings_matrix)
    
    # Save the index
    faiss.write_index(index, faiss_index_file_path)

# Initialize the dataset JSON
dataset_json = {'train': []}

# Function to apply DBSCAN filtering (replace with your actual DBSCAN function)
def dbscan_filtering(embeddings):
    return [True] * len(embeddings)  # Dummy function, replace with your actual DBSCAN logic

# For each question, find nearest contexts and filter them using DBSCAN
for i, question in enumerate(tqdm(questions, desc="Processing questions")):
    query_embedding = question_embeddings[i].reshape(1, -1)
    _, results = index.search(query_embedding, context_embeddings.shape[0])
    nearest_embeddings = context_embeddings[results[0]]
    core_samples_mask = dbscan_filtering(nearest_embeddings)
    filtered_results = [contexts[j] for j, is_core in enumerate(core_samples_mask) if is_core]
    top_5_percent = int(len(filtered_results) * 0.05)
    top_5_percent_results = filtered_results[:top_5_percent]
    
    for context in top_5_percent_results:
        dataset_json['train'].append({'context': context, 'question': question})
        print('question:\n\n', question)
        print('context:\n\n', context)

# Save the dataset to a JSON file
with open("dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)