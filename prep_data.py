import re
import random
import pandas as pd
from datasets import concatenate_datasets, load_dataset
import datasets
import pickle
import numpy as np
import os
from functions import *
from vars import *

def generate_and_store_prompts(dataset, indices, prompt_dict, prompt_templates, task_keys, key_mapping=None):
    for i in indices:
        record = dataset[i]
        
        # Prepare keys based on key_mapping
        context_key = key_mapping.get('context', 'context')
        response_key = key_mapping.get('response', 'response')
        prompt_key = key_mapping.get('prompt', 'prompt')
        
        # Get extra keys for default_prompt, context_replace_field, default_context, and append_field
        default_prompt = key_mapping.get('default_prompt', None)
        context_replace_field = key_mapping.get('context_replace_field', None)
        default_context = key_mapping.get('default_context', None)
        append_field = key_mapping.get('append_field', None)
        
        if default_prompt == 'tldr_prompt':
            default_prompt = 'TL;DR'
        elif default_prompt == 'quote_prompt':
            default_prompt = 'Provide a quote that includes'
        
        # Special handling for SQuAD 'answers' field
        response = record.get(response_key, "")
        if isinstance(response, dict) and 'text' in response:
            response = response['text'][0] if response['text'] else "N/A"
        
        # Special handling for appending a field (like 'author' to 'quote')
        if append_field:
            append_value = record.get(append_field, "")
            if append_value:
                response = f"{response} - {append_value}"
        
        context = record.get(context_key, default_context if default_context else "")
        
        # Special handling for replacing text in 'context'
        if context_replace_field:
            context = context.replace(context_replace_field, '')
        
        prompt = record.get(prompt_key, default_prompt if default_prompt else "")
        
        formatted_record = {'context': context, 'prompt': prompt, 'response': response}
        
        # Generate and store each type of prompt
        for task_key in task_keys:
            prompt_template = prompt_templates.get(task_key, None)
            if prompt_template:
                formatted_prompt = prompt_template.format(**formatted_record)
                
                if i not in prompt_dict[task_key]:
                    prompt_dict[task_key][i] = []
                
                prompt_dict[task_key][i].append(formatted_prompt)


def add_indices_to_records(records, indices):
    for i in indices:
        records[i]['index'] = i

# Function to load or download dataset
def load_or_download_dataset(pkl_path, dataset_name, splits):
    if os.path.exists(pkl_path):
        print(f"Loading {dataset_name} from pickle file...")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Downloading {dataset_name}...")
        dataset = concatenate_datasets([load_dataset(dataset_name)[split] for split in splits])
        with open(pkl_path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

#quotes_az = pd.read_csv(r"./source/quotes_by_author.csv",index_col=0)['0'].values
#quotes_gracious = pd.read_csv(r"./source/graciousquotes.csv",index_col=0)['0'].values

# Template for CQ prompts
# Standardized prompt templates
caq_prompt_template = "Context:\n\n{context}\n\nAnswer:\n\n<CAQ><preferred>{response}\n\nInstruction:\n\n{prompt}\n\n</preferred></CAQ>"
cqa_prompt_template = "Context:\n\n{context}\n\nInstruction:\n\n{prompt}\n\n<CQA><preferred>Answer:\n\n{response}\n\n</preferred></CQA>"
qca_prompt_template = "Instruction:\n\n{prompt}\n\n<QCA><preferred>Context:\n\n{context}\n\nAnswer:\n\n{response}\n\n</preferred></QCA>"
qa_prompt_template = "Instruction:\n\n{prompt}\n\n<QA><preferred>Answer:\n\n{response}\n\n</preferred></QA>"
summ_prompt_template = "Context:\n\n{context}\n\nPrompt:\n\n{prompt}\n\n<SUMM><preferred>Response:\n\n{response}\n\n</preferred></SUMM>"
quote_prompt_template = "Prompt:\n\n{quote_prompt}\n\n{tags}\n\n<QT><preferred>Response:\n\n{quote}\n\n-{author}</preferred></QT>"

caq_prompt_template = "Context:\n\n{context}\n\nAnswer:\n\n{response}\n\nInstruction:\n\n{prompt}\n\n"
cqa_prompt_template = "Context:\n\n{context}\n\nInstruction:\n\n{prompt}Answer:\n\n{response}\n\n"
qca_prompt_template = "Instruction:\n\n{prompt}Context:\n\n{context}\n\nAnswer:\n\n{response}\n\n"
qa_prompt_template = "Instruction:\n\n{prompt}Answer:\n\n{response}\n\n"
summ_prompt_template = "Context:\n\n{context}\n\nPrompt:\n\n{prompt}\n\nResponse:\n\n{response}\n\n"
quote_prompt_template = "Prompt:\n\n{quote_prompt}\n\n{tags}Response:\n\n{quote}-{author}\n\n"


# Load or download datasets_
datasets_ = {}
for info in datasets_info:
    dataset_name = info['dataset_name'].split("/")[-1]  # Extract the last part of the dataset name
    datasets_[dataset_name] = load_or_download_dataset(info['pkl_path'], info['dataset_name'], info.get('splits'))

# Access individual datasets_
squad_v2 = datasets_['squad_v2']
openai_summarize_tldr = datasets_['openai_summarize_tldr']
wizardLM = datasets_['WizardLM_evol_instruct_V2_196k']
dolly_closed_qa = datasets_['dolly_closed_qa']
dolly_15k = datasets_['databricks-dolly-15k']
english_quotes = datasets_['english_quotes']

squad_v2_indices = list(range(len(squad_v2)))  # You could customize this list as needed
dolly_closed_qa_indices = list(range(len(dolly_closed_qa)))  # You could customize this list as needed
dolly_15k_indices = list(range(len(dolly_15k)))  # You could customize this list as needed
openai_summarize_tldr_indices = list(range(len(openai_summarize_tldr)))  # You could customize this list as needed
wizardlm_train_indices = list(range(len(wizardLM)))  # You could customize this list as needed
english_quotes_indices = list(range(len(english_quotes)))  # You could customize this list as needed

# Add indices
add_indices_to_records(squad_v2, squad_v2_indices)
add_indices_to_records(dolly_closed_qa, dolly_closed_qa_indices)
add_indices_to_records(dolly_15k, dolly_15k_indices)
add_indices_to_records(openai_summarize_tldr, openai_summarize_tldr_indices)
add_indices_to_records(wizardLM, wizardlm_train_indices)
add_indices_to_records(english_quotes, english_quotes_indices)

# Initialize your dictionaries
squad_v2_prompts = {'caq': {}, 'cqa': {}, 'qca': {}, 'qa': {} }
dolly_closed_qca_prompts = {'caq': {}, 'cqa': {}, 'qca': {}, 'qa': {} }
dolly_15k_prompts = {'caq': {}, 'cqa': {}, 'qca': {}, 'qa': {} }
openai_summarize_tldr_prompts = {'summ': {} }
wizardlm_qa_prompts = {'qa': {} }
english_quotes_prompts = {'qt': {}}

# Define template dictionaries
squad_v2_templates = {'cqa': cqa_prompt_template, 'qca': qca_prompt_template, 'qa': qa_prompt_template, 'caq': caq_prompt_template}
dolly_closed_qa_templates = {'cqa': cqa_prompt_template, 'qca': qca_prompt_template, 'caq': caq_prompt_template, 'qa': qa_prompt_template}
dolly_15k_templates = {'cqa': cqa_prompt_template, 'qca': qca_prompt_template, 'caq': caq_prompt_template, 'qa': qa_prompt_template}
openai_summ_templates = {'summ': summ_prompt_template}

wizardlm_templates = {'qa': qa_prompt_template}
english_quotes_templates = {'qt': quote_prompt_template}

# Key mappings for different datasets
#alt: dynamically-derived approach based on the number of non-empty fields (lose some of the benefits of explicitness and it could become tricky to manage when you encounter datasets that don't fit neatly into those categories)
squad_v2_key_mapping = {'context': 'context', 'response': 'answers', 'prompt': 'question'}
dolly_closed_qa_key_mapping = {'context': 'context', 'response': 'response', 'prompt': 'instruction'}
dolly_15k_key_mapping = {'context': 'context', 'response': 'response', 'prompt': 'instruction'}  # Assuming same keys as dolly_closed_qa

wizardlm_key_mapping = {'context': None, 'prompt': 'instruction', 'response': 'answer'}

openai_summ_key_mapping = {
    'context': 'prompt', 
    'response': 'label', 
    'prompt': 'tldr_prompt',
    'default_prompt': 'TL;DR',
    'context_replace_field': '\nTL;DR: '
}

english_quotes_key_mapping = {
    'context': 'quote_prompt', 
    'prompt': 'tags', 
    'response': 'quote',
    'default_context': 'quote_prompt',  # Adding default_context here
    'append_field': 'author'  # Adding append_field here
}

# Existing key mappings are used here
generate_and_store_prompts(squad_v2, squad_v2_indices, squad_v2_prompts, squad_v2_templates, ['caq', 'cqa', 'qca', 'qa'], key_mapping=squad_v2_key_mapping)
generate_and_store_prompts(dolly_closed_qa, dolly_closed_qa_indices, dolly_closed_qca_prompts, dolly_closed_qa_templates, ['caq', 'cqa', 'qca', 'qa'], key_mapping=dolly_closed_qa_key_mapping)
generate_and_store_prompts(dolly_15k, dolly_15k_indices, dolly_15k_prompts, dolly_15k_templates, ['caq', 'cqa', 'qca', 'qa'], key_mapping=dolly_15k_key_mapping)
#generate_and_store_prompts(wizardLM, wizardlm_train_indices, wizardlm_qa_prompts, wizardlm_templates, ['qa'], key_mapping=wizardlm_key_mapping)
generate_and_store_prompts(openai_summarize_tldr, openai_summarize_tldr_indices, openai_summarize_tldr_prompts, openai_summ_templates, ['summ'], key_mapping=openai_summ_key_mapping)
#generate_and_store_prompts(english_quotes, english_quotes_indices, english_quotes_prompts, english_quotes_templates, ['qt'], key_mapping=english_quotes_key_mapping)

datasets_dict = {
    'squad_v2': {'pretrain': squad_v2_prompts, 'finetune': None},
    'dolly_closed_qa': {'pretrain': dolly_closed_qca_prompts, 'finetune': None},
    'dolly_15k': {'pretrain': dolly_15k_prompts, 'finetune': None},
    'openai_summarize_tldr': {'pretrain': openai_summarize_tldr_prompts, 'finetune': None},
    'wizardlm_qa': {'pretrain': wizardlm_qa_prompts},
    'english_quotes': {'pretrain': english_quotes_prompts, 'finetune': None}
}
def print_first_record_of_subdatasets(datasets_dict):
    for dataset_name, subdatasets in datasets_dict.items():
        print(f"Dataset: {dataset_name}")
        pretrain = subdatasets.get('pretrain', {})
        for subdataset_name, records in pretrain.items():
            first_record = records.get(0, [])  # Fetch the first record with index 0, if available
            if first_record:
                print(f"  Subdataset: {subdataset_name}")
                print(f"    First Record: {first_record}")
            else:
                print(f"  Subdataset: {subdataset_name} has no records.")

# Assuming datasets_dict is your dictionary containing all the datasets and their subdatasets
print_first_record_of_subdatasets(datasets_dict)
# Pickle the datasets_dict
pickle.dump(datasets_dict, open('./source/datasets_dict.pkl', 'wb'))

