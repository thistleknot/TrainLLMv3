import ctranslate2
import torch
import subprocess
import transformers
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, Trainer, EvalPrediction, TrainingArguments, TrainerControl, TrainerState, TrainerCallback, logging, pipeline, DataCollatorForLanguageModeling
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from collections import Counter
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os

#model = SentenceTransformer('all-MiniLM-L6-v2')

db_path = "conclusions.sqlite"
top_k = 10
top_n = 3
n_samples = 1000


def clear_cuda():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Function to format the sorted entity counts for the report
def format_entities_for_report(sorted_entities):
    report_lines = [f"{entity}: {count}" for entity, count in sorted_entities]
    return '\n'.join(report_lines)

# Function to save data to a JSON file
def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def replace_entities_with_representatives(data, replacement_dict):
    # Go through each item in your data
    for item in data:
        # Access the 'Premise Phase' which contains the triplets
        triplets = item['Response']['Premise Phase']

        # Go through each Triplet
        for triplet_key, triplet_value in triplets.items():
            if triplet_value:  # If the triplet is not empty
                subject = triplet_value.get('Subject')
                object_ = triplet_value.get('Object')

                # Replace subject and object with representative entities if they exist in the replacement dictionary
                if subject in replacement_dict:
                    triplet_value['Subject'] = replacement_dict[subject]
                if object_ in replacement_dict:
                    triplet_value['Object'] = replacement_dict[object_]

    return data

def load_json(file_path):
    """
    Load JSON data from a file.

    :param file_path: str, The path to the input JSON file.
    :return: dict, The data loaded from the JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Define the function to optimize DBSCAN parameters, now with an extra parameter for embeddings.
def optimize_dbscan(args, embeddings):
    eps, min_samples = args
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit and predict clusters using DBSCAN on the provided embeddings
    clusters = dbscan.fit_predict(embeddings)

    num_labels = len(set(clusters))
    # If all data points got assigned to one cluster, the silhouette score isn't meaningful.
    if num_labels == 1 or num_labels == len(embeddings):
        silhouette_avg = -1
    else:
        silhouette_avg = silhouette_score(embeddings, clusters)

    # Minimize the negative silhouette score
    return -silhouette_avg

def iterate_triplets(data):
    # Iterating through each item in your data
    entities = []
    predicates = []
    for item in data:
        # Assuming the structure is such that 'Response' is a key, and 'Premise Phase' contains the triplets
        triplets = item['Response']['Premise Phase']

        # Iterating through each Triplet in the 'Premise Phase'
        for triplet_key, triplet_value in triplets.items():
            # Checking if the triplet contains data (is not an empty string)
            if triplet_value:
                subject = triplet_value.get('Subject', 'No Subject')  # Get the subject, or a default value if it's not present
                predicate = triplet_value.get('Predicate', 'No Predicate')
                object_ = triplet_value.get('Object', 'No Object')
                # Now you can process the subject as needed
                print(f'{triplet_key} Subject:', subject)
                print(f'{triplet_key} Object:', object_)
                print(f'{triplet_key} Predicate:', predicate)
                entities.append(subject)
                #.replace('[','').replace(']','').replace('   ',' ').replace('  ',' ').replace("\n    \"",''))
                entities.append(object_)
                #.replace('[','').replace(']','').replace('   ',' ').replace('  ',' ').replace("\n    \"",''))
                predicates.append(predicate)
    return entities, predicates

def save_embeddings_to_db(embeddings):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, embedding BLOB)''')
        for emb in embeddings:
            emb_bytes = emb.tobytes()
            cursor.execute("INSERT INTO embeddings (embedding) VALUES (?)", (emb_bytes,))
        conn.commit()

def load_embeddings_from_db():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM embeddings")
        rows = cursor.fetchall()
    return [np.frombuffer(row[1], dtype=np.float32) for row in rows]

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

# Usage
if __name__ == "__main__":
    # Specify the path to your JSON file
    path_to_json = 'results.json'  # replace with your actual file path

    conclusions = []
    # Load the data from JSON file
    my_json_data = load_json(path_to_json)
    print(len(my_json_data))
    for e in my_json_data:
        if(e['Response']['Conclusion Phase'] == ''):
            print('pass')
        else:
            #print(e['Response']['Conclusion Phase'])
            conclusions.append(e['Response']['Conclusion Phase'])
    print(len(conclusions))
    selected_prompts = conclusions

    #tokenizer = transformers.AutoTokenizer.from_pretrained('/home/user/text-generation-webui/models/Llama-2-7b-hf/')

    #dataset_ = create_dataset(selected_prompts, tokenizer)
    #dataset = process_dataset(dataset_,tokenizer)

    #chunks = [tokenizer.decode(seq, skip_special_tokens=True) for seq in dataset['input_ids']]

    #print(len(chunks))
    #print(type(chunks))

    if not os.path.exists(db_path):
        # Generate embeddings if they don't exist
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Read data

        embeddings = sentence_model.encode(selected_prompts, show_progress_bar=True)
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
    samples_ = []
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
        
        selected_quote = selected_prompts[indices[0]]  # The quote corresponding to the sample
        sample_ = [(selected_prompts[idx], 1 - filtered_distances[filtered_indices.index(idx)]) for idx in selected_indices]
        final_samples_dict[selected_quote] = sample_
        #samples_.append(sample)

    promptsArray = []

    for sample, selected_matches in final_samples_dict.items():
        # Combine the matches into a single string with a newline character after each match
        context = '\n'.join(match for match, weight in selected_matches)
        promptsArray.append(context)
    
    print(len(promptsArray))
    #[print(len(p)) for p in promptsarray]

    device_map = 'cuda'

    models = {
        #'summ':"/data/text-generation-webui/models/flan-t5-3b-summarizer",
        #'summ':"/data/text-generation-webui/models/flan-t5-large-finetuned-openai-summarize_from_feedback",
        'aa':"/home/user/text-generation-webui/models/t5-large-generation-race-QuestionAnswer"#,
        #'ea':"/data/text-generation-webui/models/t5-large-generation-squad-QuestionAnswer"
        }

    for p in models:
        print(p)
        tokenizer = transformers.AutoTokenizer.from_pretrained(models[p]+'/')
        translator = ctranslate2.Translator(models[p]+'_ct/',device=device_map)

        num_q = 1

        num_c = len(promptsArray)
        bs = 16

        if(p=='summ'):
            bs = int(bs/2.6)

        device_map = 'cuda'

        print('num_c',num_c)

        input_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(c)) for c in promptsArray]

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

        #sample_size = bs*1
        sample_size = len(input_tokens)

        # Split the input_tokens into sub-batches of size 2
        for i in range(0, len(input_tokens[0:sample_size]), bs):
            if(i%100):
                print(i)

            sub_batch = input_tokens[i:i+bs]
            # Translate the sub-batch
            results = translator.translate_batch(
                sub_batch,
                max_batch_size=1,
                asynchronous=True,
                batch_type="examples",
                num_hypotheses=num_q
            )

            try:
                # Append the results to all_results
                #print([r.result() for r in results])
                all_results.extend([r.result() for r in results])
            except:
                print('error on resultsm trying sequentially',i)

                for i_ in range(0, len(sub_batch)):
                    print(len(sub_batch[i_]))

                    results = translator.translate_batch(
                        sub_batch[i_],
                        max_batch_size=1,
                        asynchronous=False,
                        batch_type="examples",
                        num_hypotheses=num_q
                    )
                    print([r.result() for r in results])
                    all_results.append([r.result() for r in results])
        print(len(all_results))

        print('converted_contexts:',len(converted_contexts))
        # Iterate through each context and its corresponding translation result
        #for idx, (context_, async_result) in enumerate(zip(converted_contexts[0:sample_size], all_results)):
        # Initialize your final output list or dictionary

        # Initialize your final output list. We won't actually use this for JSON Lines, but we will use it to keep track of objects.
        final_output = []

        # Your output file
        output_file = 'spo_qa.jsonl'

        # Open the file. Notice we're using 'w' to write text. If the file doesn't exist, it's created.
        with open(output_file, 'w') as f:

            # Iterate through each context and its corresponding translation result
            for idx, (context_, async_result) in enumerate(zip(promptsArray, all_results)):

                # Process the translation results
                for hypothesis in async_result:
                    # Decode the tokens back to text
                    output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(hypothesis['tokens']))

                    # Clean up the decoded text
                    clean_text = output_text.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")

                    # Here you should split or process the 'clean_text' to separate the 'instruction' from the 'response'
                    # Assuming the 'instruction' and 'response' are separated by some delimiter like '<sep>'
                    parts = clean_text.split('<sep>')  # Adjust based on your actual delimiter
                    if len(parts) == 2:
                        instruction, response = parts
                    else:
                        instruction = clean_text  # or some default value
                        response = ""  # or some default value

                    # Create a dictionary for the current item
                    #alpaca
                    current_item = {
                        'instruction': instruction.strip(),  # Remove leading/trailing whitespace
                        'input': context_,
                        'output': response.strip(),  # Remove leading/trailing whitespace
                        # 'category': You would need logic to determine what the 'category' is, if applicable
                    }

                    # Convert the dictionary to a JSON string and write to the file
                    f.write(json.dumps(current_item) + '\n')  # JSON object on a new line

                    # You can also keep track of this in a list if needed for something else
                    final_output.append(current_item)

                print(f'Processed {idx + 1}/{len(promptsArray)}')

        print(f'Data saved to {output_file}')

        #garbage collection
        del translator
        #clear_cuda()
