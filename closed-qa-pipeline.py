#from random import sample
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset, DatasetDict, Dataset
from datetime import datetime
from functools import partial
from loguru import logger
from peft import LoraConfig, get_peft_model, LoraModel, prepare_model_for_kbit_training, PeftModel, PeftConfig
from random import shuffle
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, Trainer, EvalPrediction, TrainingArguments, TrainerControl, TrainerState, TrainerCallback, logging, pipeline, DataCollatorForLanguageModeling
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from transformers.trainer_callback import TrainerCallback
from typing import Dict, Optional, Any, Union
import argparse
import bitsandbytes as bnb
import copy
import datasets
import datasets.distributed
import hashlib
import json
import math
import numpy as np 
import os
import pandas as pd
import pickle
import random
import tempfile
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import transformers
import wandb
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Initialize an empty list to store dataset objects
dataset = []

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
    
# Example Context
contexts = [r"""Chelsea's mini-revival continued with a third victory in a row as they consigned struggling Leicester City to a fifth consecutive defeat.
Buoyed by their Champions League win over Borussia Dortmund, Chelsea started brightly and Ben Chilwell volleyed in from a tight angle against his old club.
Chelsea's Joao Felix and Leicester's Kiernan Dewsbury-Hall hit the woodwork in the space of two minutes, then Felix had a goal ruled out by the video assistant referee for offside.
Patson Daka rifled home an excellent equaliser after Ricardo Pereira won the ball off the dawdling Felix outside the box.
But Kai Havertz pounced six minutes into first-half injury time with an excellent dinked finish from Enzo Fernandez's clever aerial ball.
Mykhailo Mudryk thought he had his first goal for the Blues after the break but his effort was disallowed for offside.
Mateo Kovacic sealed the win as he volleyed in from Mudryk's header.
The sliding Foxes, who ended with 10 men following Wout Faes' late dismissal for a second booking, now just sit one point outside the relegation zone.
""".replace('\n', ' '),
r"""World number one Novak Djokovic says he is hoping for a "positive decision" to allow him 
to play at Indian Wells and the Miami Open next month. The United States has extended 
its requirement for international visitors to be vaccinated against Covid-19. Proof of vaccination 
will be required to enter the country until at least 10 April, but the Serbian has previously 
said he is unvaccinated. The 35-year-old has applied for special permission to enter the country. 
Indian Wells and the Miami Open - two of the most prestigious tournaments on the tennis calendar 
outside the Grand Slams - start on 6 and 20 March respectively. Djokovic says he will return to 
the ATP tour in Dubai next week after claiming a record-extending 10th Australian Open title 
and a record-equalling 22nd Grand Slam men's title last month.""".replace("\n", "")]



# Load the SQuAD v2 dataset
squad_dataset = load_dataset("squad_v2")

# Extract the contexts from the dataset
train_contexts = [item['context'] for item in squad_dataset['train']]
validation_contexts = [item['context'] for item in squad_dataset['validation']]

contexts = train_contexts[20]

# Load Models and Tokenizers
qa_tokenizer = AutoTokenizer.from_pretrained("/mnt/h/models/t5-large-generation-squad-QuestionAnswer/")
model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/h/models/t5-large-generation-squad-QuestionAnswer/",quantization_config=bnb_config)

# Generate Questions and Extractive Answers
qa_inputs = qa_tokenizer(contexts, return_tensors="pt", padding=True, truncation=False)
qa_outputs = model.generate(**qa_inputs, max_length=100)
questions_answers = qa_tokenizer.batch_decode(qa_outputs, skip_special_tokens=False)
questions, extractive_answers = zip(*[qa.split('<sep>') for qa in questions_answers])

# Initialize your custom SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize BERTopic with your custom SentenceTransformer
model = BERTopic(embedding_model=sentence_model)

cleaned_extractive_answers = []

for answer in extractive_answers:
    cleaned_answer = answer.replace('</s>', '').replace('<pad>', '').strip()
    cleaned_extractive_answers.append(cleaned_answer)

print(cleaned_extractive_answers)
topics, probs = model.fit_transform(cleaned_extractive_answers)

# First, fit the topic model to all contexts

def extract_high_prob_words_for_individual_document(model, individual_document, threshold=0.5):
    """ Extract high probability words for an individual document using BERTopic """
    topics, _ = model.transform([individual_document])
    topic = topics[0]  # Assuming only one topic is relevant per document
    topic_words = model.get_topic(topic)
    
    # Filter words based on probability
    high_prob_words = [word for word, prob in topic_words if prob >= threshold]
    
    return high_prob_words

# Extract high-probability topic words for each document in cleaned_extractive_answers
nested_high_prob_words_list = [extract_high_prob_words_for_individual_document(topic_model, ea, threshold=0.5) for ea in cleaned_extractive_answers]

# Initialize your tokenizer and model for abstractive answers
aa_tokenizer = AutoTokenizer.from_pretrained("/mnt/h/models/t5-large-generation-race-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/h/models/t5-large-generation-race-QuestionAnswer", quantization_config=bnb_config)

# Create Augmented Prompts for Abstractive Answers and Generate Them
# Note that we are now using nested_high_prob_words_list instead of top_n_words_list
augmented_prompts_for_aa = [
    f"{c}\n\ntags:\n\n{t}\n\nEA:\n\n{ea}\n\n{q} <sep> answer"
    for c, t, ea, q in zip(contexts, nested_high_prob_words_list, cleaned_extractive_answers, questions)
]
aa_inputs = aa_tokenizer(augmented_prompts_for_aa, return_tensors="pt", padding=True, truncation=False)
aa_outputs = model.generate(**aa_inputs, max_length=100)
abstractive_answers = aa_tokenizer.batch_decode(aa_outputs, skip_special_tokens=True)

# Create Dataset
dataset = [
    {
        'question': q,
        'tags': t,
        'context': c,
        'extractive_answer': ea,
        'abstractive_answer': aa,
        'prompt': f"""Question:\n\n{q}\n\nTags\n\n{t}\n\nContext:\n\n{c}\n\nExtractive Answer\n\n{ea}\n\nAbstractive Answer\n\n{aa}\n\n"""
    }
    for q, t, c, ea, aa in zip(questions, nested_high_prob_words_list, contexts, cleaned_extractive_answers, abstractive_answers)
]

# Display Dataset
for i, data in enumerate(dataset):
    print(f"Dataset Entry {i+1}:\n")
    print(f"Question: {data['question']}")
    print(f"Tags: {data['tags']}")
    print(f"Context: {data['context']}")
    print(f"Extractive Answer: {data['extractive_answer']}")
    print(f"Abstractive Answer: {data['abstractive_answer']}")
    print(f"Prompt: {data['prompt']}")
    print("="*50)
