import re
import random
import pandas as pd
from datasets import concatenate_datasets, load_dataset
import datasets
import pickle
import numpy as np
import os
from functions import *

quotes_az = pd.read_csv(r"./source/quotes_by_author.csv",index_col=0)['0'].values
quotes_gracious = pd.read_csv(r"./source/graciousquotes.csv",index_col=0)['0'].values
quotes_english = load_dataset("Abirate/english_quotes")['train']['quote']

# Define file paths and dataset names
datasets_info = [
    {'pkl_path': './source/squad_v2.pkl', 'dataset_name': 'squad_v2', 'splits': ['train', 'validation']},
    {'pkl_path': './source/openai_summarize_tldr.pkl', 'dataset_name': 'CarperAI/openai_summarize_tldr', 'splits': ['train', 'valid', 'test']},
    {'pkl_path': './source/wizardLM.pkl', 'dataset_name': 'WizardLM/WizardLM_evol_instruct_V2_196k'},
    {'pkl_path': './source/dolly_closed_qa.pkl', 'dataset_name': 'lionelchg/dolly_closed_qa', 'splits': ['train', 'test']},
]

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

#squad_v2_indices = random.sample(range(sample_ratios['squad_v2']['size']), int(sample_ratios['squad_v2']['min_sample_size']))
squad_v2_indices = list(range(0,len(squad_v2)))
for i in squad_v2_indices:
    squad_v2[i]['index'] = i

#sample_ratios['squad_v2']['indices']=squad_v2_indices
squad_v2_c_prompts = [squad_v2[i]['context'] for i in squad_v2_indices]

openai_summarize_tldr_prompts = []

# For the OpenAI Summarize dataset
for i, record in enumerate(openai_summarize_tldr):
    context = record['prompt'].replace('TL;DR:','')
    tldr_prompt = 'TL;DR'
    response = record['label']
    
    #generated_prompt = generate_prompt_example(context=context, prompt=tldr_prompt, response=response, task='<SUMM>')
    generated_prompt = f"""Context:\n\n{context}\nPrompt:\n\n{tldr_prompt}\n\n<SUMM><preferred>Response:\n\n{response}\n\n</preferred><\SUMM>"""
    openai_summarize_tldr_prompts.append({'index': i, 'prompt': generated_prompt})

squad_v2_prompts = {
    'cqa': {},
    'qa': {},
    'cq': {}
}

# Loop through the SQuAD dataset
for i in squad_v2_indices:
    context = squad_v2[i]['context']
    #print(context)
    # Initialize empty lists for this index
    squad_v2_prompts['cqa'][i] = []
    squad_v2_prompts['qa'][i] = []
    squad_v2_prompts['cq'][i] = []
    
    # Now you don't need a loop here since there is only one question
    question_data = squad_v2[i]['question']
    #print('\t'+question_data)
    cq_prompt = 'Provide a question related to the context.'
    
    # For context and question (CQ)
    #cq_prompt = generate_prompt_example(context=context, prompt=cq_prompt, response=question_data, task='<CQ>')
    cq_prompt = f"""Prompt:\n\n{cq_prompt}\n\nContext:\n\n{context}\n\n<CQ><preferred>Response:\n\n{question_data}\n\n</preferred><\CQ>"""
    
    squad_v2_prompts['cq'][i].append(cq_prompt)
    
    for answer in squad_v2[i]['answers']['text']:  # assuming answers is a dict with 'text' as one of its keys
        
        # For context, question, and answer (CQA)
        #cqa_prompt = generate_prompt_example(context=context, prompt=question_data, response=answer, task='<CQA>')
        
        cqa_prompt = f"""Context:\n\n{context}\n\nPrompt:\n\n{question_data}\n\n<CQA><preferred>Response:\n\n{answer}\n\n<preferred><\CQA>"""
        squad_v2_prompts['cqa'][i].append(cqa_prompt)
        
        # For just question and answer (QA)
        #print('\t\t'+answer_text)
        qa_prompt = f"""Prompt:\n\n{question_data}\n\n<QA><preferred>Response:\n\n{answer}\n\n<preferred><\QA>"""
        #qa_prompt = generate_prompt_example(prompt=question_data, response=answer, task='<QA>')
        squad_v2_prompts['qa'][i].append(qa_prompt)

quotes = np.unique([*quotes_az, *quotes_gracious, *quotes_english])

#WizardLM

wizardlm_qa_prompts = [
    'Prompt:\n\n' + (extract_prompt_response(prompt)[0] or "") + 
    '\n\n<QA><preferred>Response:\n\n' + (extract_prompt_response(prompt)[1] or "") + 
    '\n\n<preferred><\QA>' 
    for prompt in wizardLM['train']
]

dolly_closed_qa_prompts = ['Prompt:\n\nRecall the most accurate information to answer the question.\n\n'+p['instruction']+'\n\n'+'<QCA><preferred>Context:\n\n'+p['context']+'\n\n'+'Response:\n\n'+p['response']+'\n\n<preferred><\QCA>' for p in dolly_closed_qa]

#dolly_closed_qa_prompts = ['Prompt:\n\n'+p['instruction']+'\n\n'+'<QCA><preferred>Context:\n\n'+p['context']+'\n\n'+'Response:\n\n'+p['response']+'\n\n<\QCA><preferred>' for p in dolly_closed_qa]

#dolly_closed_qa_prompts = ['Prompt:\n\nRecall the most accurate information to answer the question.\n\n'+p['instruction']+'\n\n'+'Context:\n\n'+p['context']+'\n\n'+'Response:\n\n'+p['response']+'\n\n' for p in dolly_closed_qa]

datasets_dict = {
    'squad_v2': {'pretrain': squad_v2_prompts, 'finetune': None},
    'openai_summarize_tldr': {'pretrain': openai_summarize_tldr_prompts, 'finetune': None},
    'wizardlm_qa': {'pretrain': wizardlm_qa_prompts},
    'quotes': {'pretrain': quotes, 'finetune': None},
    'dolly_closed_qa': {'pretrain': dolly_closed_qa_prompts, 'finetune': None}
}

# Pickle the datasets_dict
pickle.dump(datasets_dict, open('./source/datasets_dict.pkl', 'wb'))

