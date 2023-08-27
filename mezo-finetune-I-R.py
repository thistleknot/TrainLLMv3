#Phase III

#Good guide to base this on
#https://brev.dev/blog/fine-tuning-llama-2

from common_imports import *
from functions import *
from vars import *

os.environ['LD_LIBRARY_PATH'] += ":/home/user/env/lib/python3.11/site-packages/nvidia/cusparse/lib/"

wandb.init(mode="offline")

TASK = "text"  # Assuming you're treating this as a text generation task
TAG = "finetuning-I-R"  # Modify as per your needs

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.legacy = True
tokenizer.pad_token = tokenizer.eos_token
new_tokens = ["<preferred>", "<dispreferred>"]
special_tokens_dict = {'additional_special_tokens': new_tokens}
num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.save_pretrained('./bits-ft-I-R')

# 1. Load the text from the input file
with open('datasets_dict.pkl', 'rb') as f:
    datasets_dict = pickle.load(f)

#load from phase II
with open('random_indices.pkl', 'rb') as f:
    random_indices = pickle.load(f)

# Get records based on the selected indices
selected_records = list(dict.fromkeys([datasets_dict['squad_v2']['finetune'][i] for i in random_indices]))
pickle.dump(selected_records, open('selected_records-I-R.pkl', 'wb'))

text = EOS_TOKEN.join(selected_records)

device_map = {"": 0}

if(QUANTIZED):
		
	peft_config = PeftConfig.from_pretrained('bits-ft-pretrain')

	model = AutoModelForCausalLM.from_pretrained(
			peft_config.base_model_name_or_path,
			quantization_config=bnb_config, device_map=device_map
	)


	model.gradient_checkpointing_enable()
	model = prepare_model_for_kbit_training(model)
	model = get_peft_model(model, lora_config)

    # Resize the model embeddings to account for the new tokens
	model.resize_token_embeddings(len(tokenizer))
	model.config.use_cache = False

else:
	pass
	
train_dataset, valid_dataset, eval_subset = process_dataset(
    text=text,
    tokenizer=tokenizer,
    STRIDE_LENGTH=STRIDE_LENGTH,
    BLOCK_SIZE=BLOCK_SIZE,
    SPLIT_RATIO=SPLIT_RATIO,
    SUB_SAMPLE=SUB_SAMPLE,
    SUB_SAMPLE_RATIO=SUB_SAMPLE_RATIO,
    MIN_NUM_EVAL_EXAMPLES=MIN_NUM_EVAL_EXAMPLES,
	SHUFFLE=SHUFFLE
)

train_model(
    train_dataset=train_dataset, 
    valid_dataset=valid_dataset, 
    eval_subset=eval_subset, 
    output_dir='./bits-ft-I-R',
    BLOCK_SIZE=BLOCK_SIZE,
    GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS,
    EPOCHS=EPOCHS,
    TASK=TASK,
    MODEL_NAME=MODEL_NAME,
    TAG=TAG,
    LEARNING_RATE=LEARNING_RATE,
    WEIGHT_DECAY=WEIGHT_DECAY,
    MAX_GRAD_NORM=MAX_GRAD_NORM,
    BATCH_SIZE=BATCH_SIZE,
    OPTIM=OPTIM,
    WARM_RATIO=WARM_RATIO,
    model=model,
    tokenizer=tokenizer
)