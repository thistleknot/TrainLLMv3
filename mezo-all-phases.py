from common_imports import *
from functions import *
from vars import *
from datasets import concatenate_datasets, load_dataset
import datasets

os.environ['LD_LIBRARY_PATH'] += ":/home/user/env/lib/python3.11/site-packages/nvidia/cusparse/lib/"

tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens(special_tokens_dict)

def extract_indices(dataset):
    keys_list = list(dataset.keys())
    return range(0, len(dataset[keys_list[0]]))
    
def load_model(quantized, phase_dir, tokenizer):
    device_map = {"": 0}
        
    if quantized:
        if phase_dir:  # Ensure phase_dir is not None
            peft_config = PeftConfig.from_pretrained(phase_dir)
            model = AutoModelForCausalLM.from_pretrained(
            #model = AutoModelForMaskedLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config, 
                device_map=device_map
            )
        else:
            # Handle the case when phase_dir is None
            model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map=device_map)
            #model = AutoModelForMaskedLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map=device_map)
        
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
    else:
        # Unquantized model loading code here
        pass
    
    print('add these layers to qlora target_modules')
    print(find_target_modules(model))
    
    return model

def process_phase(phase, input_file, output_dir, phase_dir=None):
    device_map = {"": 0}
    
    TASK = "text"
    TAG = phase.replace(' ','-')
    
    if phase == "Phase I":
        selected_prompts = []
        # Loop through each file in the folder.
        for file_name in os.listdir(input_file):
            file_path = os.path.join(input_file, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    selected_prompts.append(content)
                        
    elif phase == "Phase II":
        # Load the dataset dictionary
        with open('./source/datasets_dict.pkl', 'rb') as f:
            datasets_dict = pickle.load(f)
            
            squad_v2_indices = extract_indices(datasets_dict['squad_v2']['pretrain'])
            openai_tldr_indices = extract_indices(datasets_dict['openai_summarize_tldr']['pretrain'])
            dolly_closed_qa_indices = extract_indices(datasets_dict['dolly_closed_qa']['pretrain'])
            dolly_15k_indices = extract_indices(datasets_dict['dolly_15k']['pretrain'])
            
            sampled_squad_indices = random.sample(squad_v2_indices, FINE_TUNE_SAMPLE_SIZE)
            sampled_openai_tldr_indices = random.sample(openai_tldr_indices, FINE_TUNE_SAMPLE_SIZE)
            sampled_dolly_closed_qa_indices = random.sample(dolly_closed_qa_indices, FINE_TUNE_SAMPLE_SIZE)
            sampled_dolly_15k_indices = random.sample(dolly_15k_indices, FINE_TUNE_SAMPLE_SIZE)

            # 3. Extract prompts using the sampled indices

            # For SQuAD v2
            sampled_qa_prompts = [datasets_dict['squad_v2']['pretrain']['qa'][i] for i in sampled_squad_indices]
            #not meant to be caq, some questions have non specific questions about a presumed prior context.
            sampled_caq_prompts = [datasets_dict['squad_v2']['pretrain']['caq'][i] for i in sampled_squad_indices]
            sampled_cqa_prompts = [datasets_dict['squad_v2']['pretrain']['cqa'][i] for i in sampled_squad_indices]

            # For Dolly Closed QA
            sampled_dolly_closed_qa_qa_prompts = [datasets_dict['dolly_closed_qa']['pretrain']['qa'][i] for i in sampled_dolly_closed_qa_indices]
            sampled_dolly_closed_qa_caq_prompts = [datasets_dict['dolly_closed_qa']['pretrain']['caq'][i] for i in sampled_dolly_closed_qa_indices]
            sampled_dolly_closed_qa_cqa_prompts = [datasets_dict['dolly_closed_qa']['pretrain']['cqa'][i] for i in sampled_dolly_closed_qa_indices]

            # For Dolly 15K
            sampled_dolly_15k_qa_prompts = [datasets_dict['dolly_15k']['pretrain']['qa'][i] for i in sampled_dolly_15k_indices]
            sampled_dolly_15k_caq_prompts = [datasets_dict['dolly_15k']['pretrain']['caq'][i] for i in sampled_dolly_15k_indices]
            sampled_dolly_15k_cqa_prompts = [datasets_dict['dolly_15k']['pretrain']['cqa'][i] for i in sampled_dolly_15k_indices]
            
            # For OpenAI TLDR
            sampled_openai_tldr_prompts = [datasets_dict['openai_summarize_tldr']['pretrain']['summ'][i] for i in sampled_dolly_15k_indices]
            
            selected_prompts = [\
            #*sampled_qa_prompts,\
            #*sampled_caq_prompts,\
            #*sampled_cqa_prompts,\
            *sampled_dolly_closed_qa_qa_prompts,\
            *sampled_dolly_closed_qa_caq_prompts,\
            *sampled_dolly_closed_qa_cqa_prompts,\
            #*sampled_dolly_15k_qa_prompts,\
            #*sampled_dolly_15k_caq_prompts,\
            #*sampled_dolly_15k_cqa_prompts,\
            *sampled_openai_tldr_prompts\
            ]
            pickle.dump(selected_prompts, open('./selected_prompts.pkl', 'wb'))
                
    elif phase == "Phase III":
        with open('sampled_indices.pkl', 'rb') as f:
            sampled_indices = pickle.load(f)
            sampled_squad_indices = sampled_indices[0]
        
        # Extract the corresponding 'qa' prompts
        sampled_qa_prompts = [squad_v2_prompts['qa'][i] for i in sampled_squad_indices]
        sampled_cq_prompts = [squad_v2_prompts['cq'][i] for i in sampled_squad_indices]
        
        selected_prompts = [*sampled_cq_prompts, *sampled_qa_prompts]
        
    elif phase == "Phase IV":
        with open(input_file, 'rb') as f:
            datasets_dict = pickle.load(f)
        random_indices = random.sample(range(len(datasets_dict['wizardlm_qa']['finetune'])), FINE_TUNE_SAMPLE_SIZE)
        selected_prompts = list(dict.fromkeys([datasets_dict['wizardlm_qa']['finetune'][i] for i in random_indices]))

    model = load_model(QUANTIZED, phase_dir, tokenizer)
    
    train_model(
        selected_prompts=selected_prompts,
        #train_dataset=train_dataset, 
        #valid_dataset=valid_dataset, 
        #eval_subset=eval_subset, 
        output_dir=output_dir,
        BLOCK_SIZE=BLOCK_SIZE,
        GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS,
        EPOCHS=EPOCHS,
        TASK=TASK,
        MODEL_NAME=MODEL_NAME,
        TAG=TAG,
        LEARNING_RATE=LEARNING_RATE,
        WEIGHT_DECAY=WEIGHT_DECAY,
        ADAM_BETA1=ADAM_BETA1,
        ADAM_BETA2=ADAM_BETA2,
        ADAM_EPSILON=ADAM_EPSILON,
        MAX_GRAD_NORM=MAX_GRAD_NORM,
        BATCH_SIZE=BATCH_SIZE,
        OPTIM=OPTIM,
        WARM_RATIO=WARM_RATIO,
        ZO_EPS=ZO_EPS,
        STRIDE_LENGTH=STRIDE_LENGTH,
        SPLIT_RATIO=SPLIT_RATIO,
        SUB_SAMPLE=SUB_SAMPLE,
        SUB_SAMPLE_RATIO=SUB_SAMPLE_RATIO,
        MIN_NUM_EVAL_EXAMPLES=MIN_NUM_EVAL_EXAMPLES,
        SHUFFLE=SHUFFLE, 
        lora_config=lora_config,
        model=model,
        tokenizer=tokenizer,
        bnb_config=bnb_config,
        device_map=device_map,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        mlm_prob=MLM_PROB,
        patience=PATIENCE,
		FINE_TUNE_SAMPLE_SIZE=FINE_TUNE_SAMPLE_SIZE
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    if phase in ["Phase II", "Phase III", "Phase IV"]:
        clear_model(model, output_dir)

# Execute phases
#process_phase("Phase I", INPUT_FILE, './bits')
process_phase("Phase II", 'datasets_dict.pkl', './bits-ft', './bits')
#process_phase("Phase III", 'datasets_dict.pkl', './bits-ft-I-R', 'bits-ft')
#process_phase("Phase IV", 'datasets_dict.pkl', './bits-ft-C-I-R', 'bits-ft-I-R')
