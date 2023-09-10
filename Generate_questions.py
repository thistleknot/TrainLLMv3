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


def clear_cuda():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

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

def main():
    device_map = 'cuda'

    paths = {
        'aa':"/data/text-generation-webui/models/t5-large-generation-race-QuestionAnswer",
        'ea':"/data/text-generation-webui/models/t5-large-generation-squad-QuestionAnswer"
        }

    for p in paths:
        tokenizer = transformers.AutoTokenizer.from_pretrained(paths[p]+'/')
        translator = ctranslate2.Translator(paths[p]+'_ct/',device=device_map)

        # Read data
        input_file = "/data/TrainLLMv3/input/"
        selected_prompts = read_data(input_file)

        # Create initial dataset
        dataset = process_dataset(create_dataset(selected_prompts, tokenizer),tokenizer)

        #chunking means I don't need to sort this, I can create batches of equal size.
        # Convert tokenized text to a list of strings for the API
        contexts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in dataset['input_ids']]
        print(len(contexts))

        num_q = 1

        num_c = len(contexts)

        bs = 16

        device_map = 'cuda'

        print('num_c',num_c)
        
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

        print(len(all_results))

        print('converted_contexts:',len(converted_contexts))
        # Iterate through each context and its corresponding translation result
        for idx, (context_, async_result) in enumerate(zip(converted_contexts, all_results)):

            # Initialize an empty list to hold QA pairs for the current context
            
            # Get the actual result from the AsyncTranslationResult
            for hypothesis in async_result.result():
                # Decode the tokens back to text
                output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(hypothesis['tokens']))
                
                # Split the output text into question and answer
                #question, answer = output_text.split('<sep>')
                question_answer = output_text#.split('<sep>')
                
                question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
                # Create a QA pair

                # Store the context and its QA pairs in the final_output dictionary
                # Use the index as the key and include the actual context text as part of the value
                final_output[f"{idx}"] = {
                    'context:': context_,
                    'question': question_answer.split(tokenizer.sep_token)[0],
                    'answer': '\n'.join(question_answer.split(tokenizer.sep_token)[1:]).strip()
                }
            print('final_output',len(final_output))

        # Save the final_output dictionary to an indented JSON file
        with open(f'{p}.json', 'w') as f:
            json.dump(final_output, f, indent=4)
            
        #garbage collection
        del translator
        clear_cuda()
        

main()