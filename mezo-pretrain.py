#Phase I

#Good guide to base this on
#https://brev.dev/blog/fine-tuning-llama-2

from common_imports import *
from functions import *

from vars import *

os.environ['LD_LIBRARY_PATH'] += ":/home/user/env/lib/python3.11/site-packages/nvidia/cusparse/lib/"

wandb.init(mode="offline")

TASK = "text"  # Assuming you're treating this as a text generation task
TAG = "phase-I"  # Modify as per your needs

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
new_tokens = ["<preferred>", "<dispreferred>"]
special_tokens_dict = {'additional_special_tokens': new_tokens}
num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# 1. Load the text from the input file
with open(INPUT_FILE, "r") as file:
    text = file.read()

device_map = {"": 0}

if(QUANTIZED):
    model = AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb_config, device_map=device_map)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)
    #needed when updating weights (and saving)?
    model.config.use_cache = False

else:
	#this is where unquantized model loading would go
    #load_in_8bit = quantization_config.load_in_8bit
    model = AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=quantize_config, device_map = device_map)

    #results in error 'IndexError: list index out of range' on a dataset
    #on self._batches[batch_idx].slice(i - self._offsets[batch_idx], 1) inside datasets table.py

    #model = AutoGPTQForCausalLM.from_pretrained(MODEL,quantize_config=quantize_config, device_map = device_map)
    #model = AutoGPTQForCausalLM.from_pretrained(MODEL,quantize_config=quantize_config, device_map = "auto")

    """ doesn't work with mezo 'RuntimeError: expected scalar type Half but found Float'
    # F.linear(input, self.weight, self.bias) in linear.py
    
    model = AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb_config, device_map=device_map)

    model = ReLoRaModel(
    model,
    r=128,  # replace with your desired value for lora_r
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["attn", "mlp"],
    trainable_scaling=True,  # Assuming you want to allow training of the scaling factors
    keep_original_weights=False, # Set to False to align with the assertion in ReLoRaModel
    lora_only=True,  # Train only the ReLoRa layers
    )
    """

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
    output_dir='./bits',
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
	ZO_EPS=ZO_EPS,
	EPS=EPS,
	lora_config=lora_config,
    model=model,
    tokenizer=tokenizer
)