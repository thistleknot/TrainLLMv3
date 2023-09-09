#from random import sample
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset, DatasetDict, Dataset
from datetime import datetime
from functools import partial
from loguru import logger
from peft import LoraConfig, get_peft_model, LoraModel, prepare_model_for_kbit_training, PeftModel, PeftConfig
from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
#from peft_pretraining.relora import ReLoRaModel, ReLoRaLinear
from random import shuffle
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, BitsAndBytesConfig, Trainer, EvalPrediction, TrainingArguments, TrainerControl, TrainerState, TrainerCallback, logging, pipeline, DataCollatorForLanguageModeling, StoppingCriteriaList, StoppingCriteria
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, BitsAndBytesConfig, Trainer, EvalPrediction, TrainingArguments, TrainerControl, TrainerState, TrainerCallback, logging, pipeline, DataCollatorForLanguageModeling, StoppingCriteriaList, StoppingCriteria
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