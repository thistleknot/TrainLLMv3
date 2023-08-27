from common_imports import *
from functions import *
from vars import *
from datasets import concatenate_datasets, load_dataset
import datasets

os.environ['LD_LIBRARY_PATH'] += ":/home/user/env/lib/python3.11/site-packages/nvidia/cusparse/lib/"

tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False)
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {
    'additional_special_tokens': ['<SUMM>','</SUMM>', '<CQ>','</CQ>', '<CQA>','</CQA>', '<QA>','</QA>', '<preferred>','</preferred>', '<dispreferred>','</dispreferred>'],
    'mask_token': '[MASK]'
}

datasets_info = [
    {'pkl_path': './source/squad_v2.pkl', 'dataset_name': 'squad_v2', 'splits': ['train', 'validation']},
    {'pkl_path': './source/openai_summarize_tldr.pkl', 'dataset_name': 'CarperAI/openai_summarize_tldr', 'splits': ['train', 'valid', 'test']},
    {'pkl_path': './source/wizardLM.pkl', 'dataset_name': 'WizardLM/WizardLM_evol_instruct_V2_196k'},
    {'pkl_path': './source/dolly_closed_qa.pkl', 'dataset_name': 'lionelchg/dolly_closed_qa', 'splits': ['train', 'test']},
]

num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

def clear_model(model, path):
    model.save_pretrained(path)
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

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

        # Extract squad_v2_prompts
        squad_v2_prompts = datasets_dict['squad_v2']['pretrain']

        tldr_prompts = datasets_dict['openai_summarize_tldr']['pretrain']
        tldr_indices = list(range(len(tldr_prompts)))

        # Generate a list of indices
        # Load or download datasets_
        datasets_ = {}
        for info in datasets_info:
            dataset_name = info['dataset_name'].split("/")[-1]  # Extract the last part of the dataset name
            datasets_[dataset_name] = load_or_download_dataset(info['pkl_path'], info['dataset_name'], info.get('splits'))

        # Access individual datasets
        squad_v2 = datasets_['squad_v2']
        
        dolly_closed_qa = datasets_dict['dolly_closed_qa']['pretrain']
        #print(datasets_dict.keys())
        #squad_v2_indices = random.sample(range(sample_ratios['squad_v2']['size']), int(sample_ratios['squad_v2']['min_sample_size']))
        squad_v2_indices = list(range(0,len(squad_v2)))
        dolly_closed_qa_indices = list(range(0,len(dolly_closed_qa)))
        
        #FINE_TUNE_SAMPLE_SIZE=200
        # Sample indices for fine-tuning
        sampled_squad_indices = random.sample(squad_v2_indices, FINE_TUNE_SAMPLE_SIZE)
        sampled_dolly_closed_qa_indices = random.sample(dolly_closed_qa_indices, FINE_TUNE_SAMPLE_SIZE)
        sampled_tldr_indices = random.sample(tldr_indices, FINE_TUNE_SAMPLE_SIZE)
        
        sampled_tldr_prompts = [tldr_prompts[i]['prompt'] for i in sampled_tldr_indices]
        
        sampled_dolly_closed_qa_prompts = [dolly_closed_qa[i] for i in dolly_closed_qa_indices]
        
        print('\nsampled_tldr_prompts\n')
        [print(p) for p in random.sample(sampled_tldr_prompts,3)]
        
        print('\nsampled_dolly_closed_qa_prompts\n')
        [print(p) for p in random.sample(sampled_dolly_closed_qa_prompts,3)]
        # Extract the corresponding 'cq' and 'cqa' prompts
        sampled_cqa_prompts = [j for i in sampled_squad_indices for j in squad_v2_prompts['cqa'][i] if j]
        #sampled_qa_prompts = [j for i in sampled_squad_indices for j in squad_v2_prompts['qa'][i] if j]
        sampled_cq_prompts = [j for i in sampled_squad_indices for j in squad_v2_prompts['cq'][i] if j]
        #print('\ntldr sampled_cqa_prompts\n')
        #[print(p) for p in random.sample(sampled_cqa_prompts,3)]
        #print('\nsampled_qa_prompts,3 prompts\n')
        #[print(p) for p in random.sample(sampled_qa_prompts,3)]
        print('\nsampled_cq_prompts\n')
        [print(p) for p in random.sample(sampled_cq_prompts,3)]
        # Combine the prompts for training
        #selected_prompts = [*sampled_cq_prompts, *sampled_cqa_prompts, *sampled_qa_prompts,*sampled_tldr_prompts]
        selected_prompts = [*sampled_cq_prompts,*sampled_cqa_prompts,*sampled_tldr_prompts,*sampled_dolly_closed_qa_prompts]
        pickle.dump(selected_prompts, open('selected_prompts.pkl', 'wb'))
        #selected_prompts = [*sampled_cq_prompts, *sampled_cqa_prompts, *sampled_qa_prompts]

        # Save the sampled indices for future use
        pickle.dump([sampled_squad_indices,sampled_tldr_prompts], open('sampled_indices.pkl', 'wb'))
        #pickle.dump(sampled_squad_indices, open('sampled_squad_indices.pkl', 'wb'))
        
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
        patience=PATIENCE
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
