from common_imports import *
from functions import *
from vars import *
from datasets import concatenate_datasets, load_dataset
import datasets

os.environ['LD_LIBRARY_PATH'] += ":/home/user/env/lib/python3.11/site-packages/nvidia/cusparse/lib/"

tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False)
tokenizer.add_special_tokens(special_tokens_dict)

print(tokenizer.pad_token)

device_map = {"": 0}

#inherited by phase_args
default_args = {
    'BLOCK_SIZE': BLOCK_SIZE,
    'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
    'TAG': None,
    'TASK': None,
    'EVAL_MODE': 'valid',
    'EPOCHS': EPOCHS,
    'MODEL_NAME': MODEL_NAME,
    'LEARNING_RATE': LEARNING_RATE,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'ADAM_BETA1': ADAM_BETA1,
    'ADAM_BETA2': ADAM_BETA2,
    'ADAM_EPSILON': ADAM_EPSILON,
    'MAX_GRAD_NORM': MAX_GRAD_NORM,
    'BATCH_SIZE': BATCH_SIZE,
    'WARM_RATIO': WARM_RATIO,
    'ZO_EPS': ZO_EPS,
    'STRIDE_LENGTH': STRIDE_LENGTH,
    'SPLIT_RATIO': SPLIT_RATIO,
    'SUB_SAMPLE': SUB_SAMPLE,
    'SUB_SAMPLE_RATIO': SUB_SAMPLE_RATIO,
    'MIN_NUM_EVAL_EXAMPLES': MIN_NUM_EVAL_EXAMPLES,
    'SHUFFLE': SHUFFLE,
    'mlm_prob': MLM_PROB,
    'patience': PATIENCE,
    'FINE_TUNE_SAMPLE_SIZE': FINE_TUNE_SAMPLE_SIZE,
    'EVAL_METRIC': 'eval',
    'min_epochs': min_epochs
}


def load_model(quantized, prior_phase_dir, tokenizer):
    device_map = {"": 0}
        
    if quantized:
        if prior_phase_dir:  # Ensure prior_phase_dir is not None
            peft_config = PeftConfig.from_pretrained(prior_phase_dir)
            model = AutoModelForCausalLM.from_pretrained(
            #model = AutoModelForMaskedLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config, 
                device_map=device_map
            )
        else:
            # Handle the case when prior_phase_dir is None
            model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map=device_map)
            #model = AutoModelForMaskedLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map=device_map)
        
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)
        model.config.use_cache = False
    else:
        # Unquantized model loading code here
        pass
    
    print('add these layers to qlora target_modules')
    print(find_target_modules(model))
    
    return model

def process_phase(phase, output_dir, prior_phase_dir=None):
    phase_args = default_args.copy()
    TASK = phase
    phase_args['TASK'] = TASK
    
    TAG = phase.replace(' ','-')
    phase_args['TAG'] = TAG
    phase_args['WARM_RATIO'] = WARM_RATIO
    #phase_args['EVAL_MODE'] = 'train'
    
    if phase == "Phase I":
        
        selected_prompts = []
        # Loop through each file in the folder.
        """
        for file_name in os.listdir(input_file):
            file_path = os.path.join(input_file, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    selected_prompts.append(content)
        """
        selected_prompts = [\
        #*sampled_qa_prompts,\
        #*sampled_caq_prompts,\
        #*sampled_cqa_prompts,\
        
		#*sampled_dolly_closed_qa_qa_prompts,\
        #*sampled_dolly_closed_qa_caq_prompts,\
        
		*sampled_dolly_closed_qa_cqa_prompts,\
        #*sampled_dolly_closed_qa_qca_prompts,\
        
        #*sampled_dolly_15k_qa_prompts,\
        #*sampled_dolly_15k_caq_prompts,\
        #*sampled_dolly_15k_cqa_prompts,\
        #*sampled_openai_tldr_prompts\
        ]
        #[print(p) for p in selected_prompts]
        pickle.dump(selected_prompts, open('./selected_prompts_I.pkl', 'wb'))
                        
    elif phase == "Phase II":

        # Load the dataset dictionary
        selected_prompts = [\
        #*sampled_qa_prompts,\
        #*sampled_caq_prompts,\
        #*sampled_cqa_prompts,\
        *sampled_dolly_closed_qa_qa_prompts,\
        #*sampled_dolly_closed_qa_caq_prompts,\
        #*sampled_dolly_closed_qa_cqa_prompts,\
        #*sampled_dolly_closed_qa_qca_prompts,\
        #*sampled_dolly_15k_qa_prompts,\
        #*sampled_dolly_15k_caq_prompts,\
        #*sampled_dolly_15k_cqa_prompts,\
        #*sampled_openai_tldr_prompts\
        ]
        pickle.dump(selected_prompts, open('./selected_prompts_II.pkl', 'wb'))

    elif phase == "Phase III":

        selected_prompts = [\
        #*sampled_qa_prompts,\
        #*sampled_caq_prompts,\
        #*sampled_cqa_prompts,\
        #*sampled_dolly_closed_qa_qa_prompts,\
        *sampled_dolly_closed_qa_caq_prompts,\
        #*sampled_dolly_closed_qa_cqa_prompts,\
        #*sampled_dolly_closed_qa_qca_prompts,\
        #*sampled_dolly_15k_qa_prompts,\
        #*sampled_dolly_15k_caq_prompts,\
        #*sampled_dolly_15k_cqa_prompts,\
        #*sampled_openai_tldr_prompts\
        ]
        pickle.dump(selected_prompts, open('./selected_prompts_III.pkl', 'wb'))

    elif phase == "Phase IV":
        phase_args['EVAL_METRIC'] = 'cosine'

        selected_prompts = [\
        #*sampled_qa_prompts,\
        #*sampled_caq_prompts,\
        #sampled_qca_prompts,\
        #*sampled_dolly_closed_qa_qa_prompts,\
        #*sampled_dolly_closed_qa_caq_prompts,\
        #*sampled_dolly_closed_qa_cqa_prompts,\
        *sampled_dolly_closed_qa_qca_prompts,\
        #*sampled_dolly_15k_qa_prompts,\
        #*sampled_dolly_15k_caq_prompts,\
        #*sampled_dolly_15k_cqa_prompts,\
        #*sampled_openai_tldr_prompts\
        ]
        pickle.dump(selected_prompts, open('./selected_prompts_IV.pkl', 'wb'))

    model = load_model(QUANTIZED, prior_phase_dir, tokenizer)
    
    train_model(
        selected_prompts=selected_prompts,
        output_dir=output_dir,
        prior_phase_dir=prior_phase_dir,
        lora_config=lora_config,
        model=model,
        tokenizer=tokenizer,
        bnb_config=bnb_config,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        OPTIM=OPTIM,
        device_map=device_map,
        #phase=phase,
        **phase_args  # unpack the other args here
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    if phase in ["Phase II", "Phase III", "Phase IV"]:
        clear_model(model, output_dir)

with open('./source/datasets_dict.pkl', 'rb') as f:
    datasets_dict = pickle.load(f)
    
    #squad_v2_indices = extract_indices(datasets_dict['squad_v2']['pretrain'])
    #openai_tldr_indices = extract_indices(datasets_dict['openai_summarize_tldr']['pretrain'])
    #dolly_closed_qa_indices = extract_indices(datasets_dict['dolly_closed_qa']['pretrain'])
    dolly_15k_indices = extract_indices(datasets_dict['dolly_15k']['pretrain'])
    
    #sampled_squad_indices = random.sample(squad_v2_indices, FINE_TUNE_SAMPLE_SIZE)
    #sampled_openai_tldr_indices = random.sample(openai_tldr_indices, FINE_TUNE_SAMPLE_SIZE)
    #sampled_dolly_closed_qa_indices = random.sample(dolly_closed_qa_indices, FINE_TUNE_SAMPLE_SIZE)
    sampled_dolly_15k_indices = random.sample(dolly_15k_indices, FINE_TUNE_SAMPLE_SIZE)

    # 3. Extract prompts using the sampled indices
    """
    # For SQuAD v2
    sampled_qa_prompts = [datasets_dict['squad_v2']['pretrain']['qa'][i] for i in sampled_squad_indices]
    #not meant to be caq, some questions have non specific questions about a presumed prior context.
    sampled_caq_prompts = [datasets_dict['squad_v2']['pretrain']['caq'][i] for i in sampled_squad_indices]
    sampled_cqa_prompts = [datasets_dict['squad_v2']['pretrain']['cqa'][i] for i in sampled_squad_indices]
    sampled_qca_prompts = [datasets_dict['squad_v2']['pretrain']['qca'][i] for i in sampled_squad_indices]

    # For Dolly Closed QA
    sampled_dolly_closed_qa_qa_prompts = [datasets_dict['dolly_closed_qa']['pretrain']['qa'][i] for i in sampled_dolly_closed_qa_indices]
    sampled_dolly_closed_qa_caq_prompts = [datasets_dict['dolly_closed_qa']['pretrain']['caq'][i] for i in sampled_dolly_closed_qa_indices]
    sampled_dolly_closed_qa_cqa_prompts = [datasets_dict['dolly_closed_qa']['pretrain']['cqa'][i] for i in sampled_dolly_closed_qa_indices]
    sampled_dolly_closed_qa_qca_prompts = [datasets_dict['dolly_closed_qa']['pretrain']['qca'][i] for i in sampled_dolly_closed_qa_indices]
    """
    # For Dolly 15K
    sampled_dolly_15k_qa_prompts = [datasets_dict['dolly_15k']['pretrain']['qa'][i] for i in sampled_dolly_15k_indices]
    sampled_dolly_15k_caq_prompts = [datasets_dict['dolly_15k']['pretrain']['caq'][i] for i in sampled_dolly_15k_indices]
    sampled_dolly_15k_cqa_prompts = [datasets_dict['dolly_15k']['pretrain']['cqa'][i] for i in sampled_dolly_15k_indices]
    print(len(datasets_dict['dolly_15k']['pretrain']['qca']))
    
    sampled_dolly_15k_qca_prompts = [datasets_dict['dolly_15k']['pretrain']['qca'][i] for i in sampled_dolly_15k_indices]
    print(len(sampled_dolly_15k_qca_prompts))
    
    # For OpenAI TLDR
    sampled_openai_tldr_prompts = [datasets_dict['openai_summarize_tldr']['pretrain']['summ'][i] for i in sampled_dolly_15k_indices]

# Execute phases
process_phase("Phase I", output_dir='./bits', prior_phase_dir=None)
#process_phase("Phase II", output_dir='./bits-ft', prior_phase_dir='./bits')
#process_phase("Phase III", output_dir='./bits-ft-I-R', prior_phase_dir='bits-ft')
#process_phase("Phase IV", output_dir='./bits-ft-C-I-R', prior_phase_dir='bits-ft-I-R')

