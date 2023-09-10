import json
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from ast import literal_eval
import torch
from peft import LoraModel, LoraConfig

with open("resources/train.json", "r") as f:
    args = json.load(f)

BLOCK_SIZE = args['block_size']
SUB_SAMPLE_RATIO = args['sub_sample_ratio']
SUB_SAMPLE = args['sub_sample']
BATCH_SIZE = args['batch_size']
STRIDE_LENGTH = int(BLOCK_SIZE*args['stride_ratio'])
MODEL_NAME = args['model_name']
MODEL = args['model']
INPUT_FILE = args['input_file']
SHUFFLE = args['shuffle']
SPLIT_RATIO = args['split_ratio']
GRADIENT_ACCUMULATION_STEPS = args['gradient_accumulation_steps']
EPOCHS = args['epochs']
LEARNING_RATE = args['learning_rate']
WARM_RATIO = args['warm_ratio']
WEIGHT_DECAY = args['weight_decay']
ADAM_BETA1 = args['adam_beta1']
ADAM_BETA2 = args['adam_beta2']
ADAM_EPSILON = args['adam_epsilon']
MAX_GRAD_NORM = args['max_grad_norm']
GROUP_SIZE = args['group_size']
MIN_NUM_EVAL_EXAMPLES = args['num_eval_examples']
SEED = args['seed']
OPTIM = args['optim']
QUANTIZED = args['quantized']
FINE_TUNE_SAMPLE_SIZE = args['fine_tune_sample_size']
EOS_TOKEN = args['eos_token']
ZO_EPS = args['zo_eps']
LR_SCHEDULER_TYPE = args['lr_scheduler_type']
MLM_PROB = args['mlm_prob']
PATIENCE = args['patience']
min_epochs = args['min_epochs']

#print(ADAM_BETA1)
quantize_config = BaseQuantizeConfig(
    bits=2,  # quantize model to 4-bit
    group_size=GROUP_SIZE,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args["load_in_4bit"],
    bnb_4bit_use_double_quant=args["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=args["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=eval(args["bnb_4bit_compute_dtype"])
)

lora_config = LoraConfig(
    r=args["lora_r"],
    lora_alpha=args["lora_alpha"],
    lora_dropout=args["lora_dropout"],
    bias=args["bias"],
    task_type=args["lora_task_type"],
    target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"
      ]
)

special_tokens_dict = {
    'additional_special_tokens': ['<SUMM>','</SUMM>', '<CQ>','</CQ>', '<QCA>','</QCA>','<CQA>','</CQA>', '<QA>','</QA>', '<QT>','</QT>', '<preferred>','</preferred>', '<dispreferred>','</dispreferred>'],
    'mask_token': '<MASK>',
    'pad_token': '<PAD>'
}

datasets_info = [
    {'pkl_path': './source/squad_v2.pkl', 'dataset_name': 'squad_v2', 'splits': ['train', 'validation']},
    {'pkl_path': './source/openai_summarize_tldr.pkl', 'dataset_name': 'CarperAI/openai_summarize_tldr', 'splits': ['train', 'valid', 'test']},
    {'pkl_path': './source/wizardLM.pkl', 'dataset_name': 'WizardLM/WizardLM_evol_instruct_V2_196k', 'splits': ['train']},
    {'pkl_path': './source/dolly_closed_qa.pkl', 'dataset_name': 'lionelchg/dolly_closed_qa', 'splits': ['train', 'test']},
    {'pkl_path': './source/dolly-15k.pkl', 'dataset_name': 'databricks/databricks-dolly-15k', 'splits': ['train']},
    {'pkl_path': './source/english_quotes.pkl', 'dataset_name': 'Abirate/english_quotes', 'splits': ['train']},
    {'pkl_path': './source/ea_zeyl.pkl', 'dataset_name': 'thistleknot/ea_zeyl', 'splits': ['train']},
    {'pkl_path': './source/aa_zeyl.pkl', 'dataset_name': 'thistleknot/aa_zeyl', 'splits': ['train']}
]

