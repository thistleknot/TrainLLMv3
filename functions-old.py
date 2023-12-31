from common_imports import *
import time
from torch import nn
from packaging import version
import inspect
from dataclasses import dataclass, is_dataclass, asdict
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import shutil
from torch.utils.data import SequentialSampler

def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]
            
            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    # default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)

from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    is_accelerate_available,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

def shuffle_dataset(dataset_dict):
    indices = list(range(len(next(iter(dataset_dict.values())))))
    random.shuffle(indices)

    shuffled_dataset = {}
    for key, values in dataset_dict.items():
        shuffled_dataset[key] = [values[i] for i in indices]

    return shuffled_dataset

def process_dataset(dataset_dict, tokenizer, STRIDE_LENGTH, BLOCK_SIZE, SPLIT_RATIO, SUB_SAMPLE, SUB_SAMPLE_RATIO, MIN_NUM_EVAL_EXAMPLES, SHUFFLE):
    
    eos_token_id = tokenizer.eos_token_id
    tokenized_text = torch.tensor([token for sublist in dataset_dict["input_ids"] for token in (sublist + [eos_token_id])])[:-1]
    total_length = len(tokenized_text)

    # 2. Generate stride lengths dynamically
    stride_lengths = [STRIDE_LENGTH]
    while stride_lengths[-1] > 1:
        stride_lengths.append(stride_lengths[-1] // 2)

    # 3. Choose the stride length with the highest modulus when divided by total_length
    STRIDE_LENGTH = max(stride_lengths, key=lambda x: total_length % x)
    print("Optimally derived STRIDE_LENGTH", STRIDE_LENGTH)

    # 3. Split the tokenized text into sequences of length `block_size`
    input_ids_list = []

    for i in range(0, len(tokenized_text), STRIDE_LENGTH):
        end = i + BLOCK_SIZE
        partial_sequence = tokenized_text[i:min(end, len(tokenized_text))]

        # If we're at the last chunk and it's shorter than BLOCK_SIZE
        if end >= len(tokenized_text):
            num_padding = BLOCK_SIZE - len(partial_sequence)
            padding = [eos_token_id] * num_padding
            partial_sequence = torch.cat([partial_sequence, torch.tensor(padding, dtype=torch.long)], dim=0)

        input_ids_list.append(partial_sequence)

    # 4. Create attention masks
    attention_mask_list = [[1] * BLOCK_SIZE for _ in input_ids_list]

    # 5. Use the same tokenized sequences for labels
    labels_list = input_ids_list.copy()
    input_ids = [seq.tolist() for seq in input_ids_list]
    attention_mask = attention_mask_list
    labels = [seq.tolist() for seq in labels_list]
    
    print('total_length', total_length)
    
    train_input_ids, valid_input_ids, train_attention_mask, valid_attention_mask, train_labels, valid_labels = train_test_split(
        input_ids, attention_mask, labels, test_size=SPLIT_RATIO, shuffle=SHUFFLE)

    train_lengths = [len(seq) for seq in train_input_ids]
    valid_lengths = [len(seq) for seq in valid_input_ids]
    train_unique_lengths = set(train_lengths)
    valid_unique_lengths = set(valid_lengths)
    print(train_unique_lengths, valid_unique_lengths)

    train_dataset = datasets.Dataset.from_dict({
        "input_ids": train_input_ids,
        "attention_mask": train_attention_mask,
        "labels": train_labels
    })

    valid_dataset = datasets.Dataset.from_dict({
        "input_ids": valid_input_ids,
        "attention_mask": valid_attention_mask,
        "labels": valid_labels
    })

    print(len(train_dataset), len(valid_dataset))

    if (SUB_SAMPLE==True):
        train_dataset = train_dataset.select(list(range(int(len(train_dataset) * SUB_SAMPLE_RATIO))))
        valid_dataset = valid_dataset.select(list(range(int(len(valid_dataset) * SUB_SAMPLE_RATIO))))
        eval_subset = subsample_dataset(valid_dataset, fraction=0.05)
        if len(eval_subset) < MIN_NUM_EVAL_EXAMPLES:
            eval_subset = valid_dataset.select(range(1))
        if len(eval_subset) > 4:
            eval_subset = eval_subset.select(range(4))
        #eval_subset = valid_dataset
    else:
        #leave train/valid alone
        eval_subset = subsample_dataset(valid_dataset, fraction=0.05)
        if len(eval_subset) < MIN_NUM_EVAL_EXAMPLES:
            eval_subset = valid_dataset.select(range(1))
        if len(eval_subset) > 4:
            eval_subset = eval_subset.select(range(4))

    print(len(train_dataset), len(valid_dataset))
    return train_dataset, valid_dataset, eval_subset

def create_dataset(text_list, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for text in text_list:
        tokenized_text = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').squeeze()
        attention_mask = [1] * len(tokenized_text) # Adjusting for variable lengths
        input_ids_list.append(tokenized_text.tolist())  # Truncate or pad as needed
        
        attention_mask_list.append(attention_mask)
        labels_list.append(tokenized_text.tolist())  # Using the same tokenized sequence for labels

    dataset_dict = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }
    
    return dataset_dict

def train_model(selected_prompts, output_dir, BLOCK_SIZE, GRADIENT_ACCUMULATION_STEPS, EPOCHS, TASK, MODEL_NAME, TAG, LEARNING_RATE, WEIGHT_DECAY, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, MAX_GRAD_NORM, BATCH_SIZE, OPTIM, WARM_RATIO, ZO_EPS, STRIDE_LENGTH, SPLIT_RATIO, SUB_SAMPLE, SUB_SAMPLE_RATIO, MIN_NUM_EVAL_EXAMPLES, SHUFFLE, lora_config, model, tokenizer, bnb_config, device_map, lr_scheduler_type, mlm_prob, patience):
    
    dataset_ = create_dataset(selected_prompts, tokenizer)
    
    """
    assert all(len(dataset_["input_ids"][i]) == len(dataset_["attention_mask"][i]) == len(dataset_["labels"][i]) for i in range(len(dataset_)))
    assert all(isinstance(dataset_["input_ids"][i], torch.Tensor) for i in range(len(dataset_)))
    assert all(isinstance(dataset_["attention_mask"][i], torch.Tensor) for i in range(len(dataset_)))
    assert all(isinstance(dataset_["labels"][i], torch.Tensor) for i in range(len(dataset_)))
    sample_indexes = random.sample(range(len(dataset_)), 5)  # Pick 5 random samples
    for index in sample_indexes:
        decoded_text = tokenizer.decode(dataset_["input_ids"][index])
        print("Original Text: ", records[index])
        print("Decoded Text: ", decoded_text)
        print("---------")
    print("Total records in dataset_:", len(dataset_))
    print("Minimum sequence length:", min(len(dataset_["input_ids"][i]) for i in range(len(dataset_))))
    print("Maximum sequence length:", max(len(dataset_["input_ids"][i]) for i in range(len(dataset_))))        

    for i in range(len(dataset_)):
        assert all(val == 1 for val in dataset_["attention_mask"][i])

    for i in range(len(dataset_)):
        assert torch.equal(dataset_["input_ids"][i], dataset_["labels"][i])
    """
    patience_counter = 0
    best_eval_perplexity = float('inf')  # start with a high value
    recent_perplexities = []
    print("LEARNING_RATE:",LEARNING_RATE)
    
    for i in range(EPOCHS):
        print('epoch:',i)
        
        dataset_ = shuffle_dataset(dataset_)
        
        train_dataset, valid_dataset, eval_subset = process_dataset(
            dataset_dict=dataset_,
            tokenizer=tokenizer,
            STRIDE_LENGTH=STRIDE_LENGTH,
            BLOCK_SIZE=BLOCK_SIZE,
            SPLIT_RATIO=SPLIT_RATIO,
            SUB_SAMPLE=SUB_SAMPLE,
            SUB_SAMPLE_RATIO=SUB_SAMPLE_RATIO,
            MIN_NUM_EVAL_EXAMPLES=MIN_NUM_EVAL_EXAMPLES,
            SHUFFLE=SHUFFLE
        )
        
        # Get number of sequences for each split
        num_train_sequences = len(train_dataset)
        num_valid_sequences = len(valid_dataset)

        # Calculate epoch steps for each split
        train_epoch_steps = num_train_sequences / BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        valid_epoch_steps = num_valid_sequences / BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

        # Calculate max training steps
        #max_train_steps = int(train_epoch_steps * EPOCHS)
        max_train_steps = int(train_epoch_steps * 1)

        min_steps = int(train_epoch_steps*1)

        #print('perplexity starts counting at:', min_steps, ' eval:', EVAL_STEPS)

        #early_stopping_callback = EarlyStoppingCallback(patience=2, min_perplexity=90, min_steps=min_steps)
    
        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
            
        if(i>=1):
            #skip warmup
            print('skip warmup')
            trainer_args = create_arguments(
                task=TASK + "-train"+str(i)+"-",
                model_name=MODEL_NAME,
                tag=TAG,
                learning_rate=eval_trainer.optimizer.param_groups[0]['lr'],
                max_steps=max_train_steps,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                weight_decay=WEIGHT_DECAY,
                adam_beta1=ADAM_BETA1,
                adam_beta2=ADAM_BETA2,
                adam_epsilon=ADAM_EPSILON,
                max_grad_norm=MAX_GRAD_NORM,
                batch_size=BATCH_SIZE,
                optim=OPTIM,
                warm_ratio=0,
                lora_config=lora_config,
                block_size=BLOCK_SIZE,
                zo_eps=ZO_EPS,
                lr_scheduler_type=lr_scheduler_type,
                data_collator=data_collator
    
            )
            
            trainer = initialize_trainer(training_args, eval_trainer.model, train_dataset, eval_subset, tokenizer)
            
            trainer.optimizer = eval_trainer.optimizer
            trainer.lr_scheduler = eval_trainer.lr_scheduler
            
        else:
            print('warmup',WARM_RATIO)
            #Warmup
            training_args = create_arguments(
                task=TASK + "-train"+str(i)+"-",
                model_name=MODEL_NAME,
                tag=TAG,
                learning_rate=LEARNING_RATE,
                max_steps=max_train_steps,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                weight_decay=WEIGHT_DECAY,
                adam_beta1=ADAM_BETA1,
                adam_beta2=ADAM_BETA2,
                adam_epsilon=ADAM_EPSILON,
                max_grad_norm=MAX_GRAD_NORM,
                batch_size=BATCH_SIZE,
                optim=OPTIM,
                warm_ratio=WARM_RATIO,
                train_epoch_steps=train_epoch_steps,
                lora_config=lora_config,
                block_size=BLOCK_SIZE,
                zo_eps=ZO_EPS,
                lr_scheduler_type=lr_scheduler_type,
                data_collator=data_collator
            )
            
        if(i==0):
            print("num_train_sequences:", num_train_sequences)
            print("num_valid_sequences:", num_valid_sequences)
            print("BLOCK_SIZE:", BLOCK_SIZE)
            print("BATCH_SIZE:", BATCH_SIZE)
            print("GRADIENT_ACCUMULATION_STEPS:", GRADIENT_ACCUMULATION_STEPS)
            print("train_epoch_steps:", train_epoch_steps)
            print("valid_epoch_steps:", valid_epoch_steps)
            print('num_train_sequences:', num_train_sequences)
            print('max_train_steps:', max_train_steps)
            print('epochs:', EPOCHS)
            print('train_epoch_steps:', train_epoch_steps)
            print("Training dataset size:", len(train_dataset))
            print("Validation dataset size:", len(valid_dataset))
            print("Evaluation subset size:", len(eval_subset))

            trainer = initialize_trainer(training_args, model, train_dataset, eval_subset, tokenizer)
        
        wandb.init(mode="offline")
        print('model_id:',id(trainer.model))
        trainer.train()
        wandb.finish()

        #number_of_steps = trainer.state.global_step
        
        # end on epochs
        EVAL_STEPS = trainer.state.global_step
        # EVAL_STEPS = len(train_dataset)/(BS*GAS)  # Modify as per your needs

        # Continue training on evaludation data at least learning_rate (no warmup)
        
        #no warmup
        valid_args = create_arguments(
            task=TASK + "-valid"+str(i)+"-",
            model_name=MODEL_NAME,
            tag=TAG,
            learning_rate=trainer.optimizer.param_groups[0]['lr'],
            max_steps=EVAL_STEPS,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            weight_decay=WEIGHT_DECAY,
            adam_beta1=ADAM_BETA1,
            adam_beta2=ADAM_BETA2,
            adam_epsilon=ADAM_EPSILON,
            max_grad_norm=MAX_GRAD_NORM,
            batch_size=BATCH_SIZE,
            optim=OPTIM,
            warm_ratio=0,
            lora_config=lora_config,
            block_size=BLOCK_SIZE,
            zo_eps=ZO_EPS,
            lr_scheduler_type=lr_scheduler_type,
            data_collator=data_collator
        )

        eval_trainer = initialize_trainer(valid_args, trainer.model, valid_dataset, eval_subset, tokenizer)
        print('model_id:',id(eval_trainer.model))   
        eval_trainer.optimizer = trainer.optimizer
        eval_trainer.lr_scheduler = trainer.lr_scheduler

        wandb.init(mode="offline")
        eval_trainer.train()
        # Check eval_loss for early stopping
            
        eval_loss = eval_trainer.evaluate()["eval_loss"]
        eval_perplexity = np.exp(eval_loss)
        
        recent_perplexities.append(eval_perplexity)
        std_dev = np.std(recent_perplexities)

        if eval_perplexity < best_eval_perplexity:
            print('perplexity improved')
            best_eval_perplexity = eval_perplexity
            patience_counter = 0
            # Save the best model
            eval_trainer.save_model(output_dir)
        else:
            print('perplexity decreased')
            patience_counter += 1

        wandb.finish()  # Move this here, after evaluation and logging operations

        if patience_counter >= patience:
            print("Early stopping triggered after", i, "epochs.")
            # Load the configuration first
            peft_config = PeftConfig.from_pretrained(output_dir)
            
            # Load the best model
            model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config, 
                device_map=device_map
            )
            break

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_perplexity=100, min_steps=None):
        super().__init__()
        self.patience = patience
        self.min_perplexity = min_perplexity
        self.min_steps = min_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'eval_loss' not in logs or state.global_step < self.min_steps:
            return control

        # Calculate perplexity from loss
        eval_perplexity = np.exp(logs.get('eval_loss'))

        if state.best_metric is None or eval_perplexity < state.best_metric:
            state.best_metric = eval_perplexity

            # Check if last_checkpoint exists
            if hasattr(state, 'last_checkpoint') and state.last_checkpoint:
                state.best_model_checkpoint = state.last_checkpoint
                # Save the model since eval_loss improved
                self._save_model(state.last_checkpoint)
            else:
                # Handle the scenario where there's no checkpoint yet
                # For now, we're just continuing without doing anything,
                # but you can add logic here if needed
                pass

            state.patience_counter = 0
        else:
            state.patience_counter += 1
            if state.patience_counter >= self.patience:
                if eval_perplexity > self.min_perplexity:
                    control.should_training_stop = True

        return control

    def _save_model(self, output_dir):
        # Use the Trainer's save_model method to save the model
        self.trainer.save_model(output_dir)

def create_arguments(task, model_name, tag, learning_rate, max_steps, gradient_accumulation_steps, weight_decay, adam_beta1, adam_beta2, adam_epsilon, max_grad_norm, batch_size, optim, block_size, zo_eps, lr_scheduler_type, data_collator, lora_config=None, warm_ratio=None, train_epoch_steps=None):
    """Generate OurArguments object."""
    lora = True if lora_config else False
    lora_alpha = lora_config.lora_alpha if lora_config else 32
    lora_r = lora_config.r if lora_config else 8
    lora_dropout = lora_config.lora_dropout if lora_config else 0.05
    bias = lora_config.bias if lora_config else None
    warmup = int(train_epoch_steps *
                 warm_ratio) if warm_ratio and train_epoch_steps else 0
    print('train_epoch_steps:',train_epoch_steps)
    print('warm_ratio:',warm_ratio)
    print('warmup:',warmup)
    return OurArguments(
        lora=lora,
        lora_alpha=lora_alpha,
        r=lora_r,
        bias=bias,
        per_device_train_batch_size=batch_size,
        max_length=block_size,
        learning_rate=learning_rate,
        logging_dir=f'./logs/{task}_{model_name}-{tag}',
        logging_steps=1,
        max_steps=max_steps,
        evaluation_strategy="steps",
        save_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        push_to_hub=False,
        output_dir=f"./result/{task}_{model_name}-{tag}",
        overwrite_output_dir=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        #warmup_steps=warmup,
        warm_ratio=warm_ratio,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        optim=optim,
        zo_eps=zo_eps,
        data_collator=data_collator
    )


def initialize_trainer(args, model, train_dataset, eval_dataset, tokenizer, callbacks=None):
    """Initialize and return MeZOTrainer."""

    return MeZOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #callbacks=callbacks,
        tokenizer=tokenizer
    )

def subsample_dataset(dataset, fraction=0.1):
    """Subsample a given fraction of the dataset."""
    num_samples = len(dataset)
    subsample_size = int(fraction * num_samples)

    # Randomly choose indices without replacement
    subsample_indices = np.random.choice(
        num_samples, subsample_size, replace=False)
    return dataset.select(subsample_indices)

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    data_collator: DataCollatorForLanguageModeling = None
    # Number of examples
    # ICL mode: number of demonstrations; training mode: number of training samples
    num_train: int = 0
    # (only enabled with training) number of development samples
    num_dev: int = None
    num_eval: int = None  # number of evaluation samples
    num_train_sets: int = None  # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None  # designated seed to sample training samples/demos
    # file name for saving performance; if None, then use the task name, model name, and config
    result_file: str = None
    #fp16: bool = True
    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    # do not load model by auto device; should turn this on when using FSDP
    no_auto_device: bool = False
    warm_ratio: float = 0.5
    #warmup_steps: int = 0
    # Calibration
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none"
    # options
    # - none: no training -- for zero-shot or in-context learning (ICL)
    # - regular: regular huggingface trainer -- for fine-tuning
    # - zo: zeroth-order (MeZO) training
    only_train_option: bool = True  # whether to only train the option part of the input
    # take the log likelihood of all options and train as classification
    train_as_classification: bool = False

    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use reparameterization trick
    # initialize prefix by real activations of random words
    prefix_init_by_real_act: bool = True

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 32  # alpha in LoRA
    lora_dropout: float = 0.05,
    r: int = 8  # r in LoRA
    bias: str = None

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    # linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    # use non-differentiable objective (only support F1 for SQuAD for now)
    non_diff: bool = False

    # Auto saving when interrupted
    # save model when interrupted (useful for long training)
    save_on_interrupt: bool = False


class MeZOTrainer(Trainer):

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.train_sampler = SequentialSampler(self.train_dataset)
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * \
            args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    # MeZO added: estimate gradient
                    tr_loss_step = self.zo_step(model, inputs)
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / \
                        (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (
                        step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(
                            accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(
                            True)
                    # MeZO added: update model with the estimated gradient
                    self.zo_update(model)

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                            self.optimizer.step()
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + \
                        (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control)

                    self._maybe_log_save_evaluate(
                        tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(
            random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(
            ), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            print("Inputs to the model:", inputs)
            outputs = model(**inputs)
            print("Outputs from the model:", outputs)

            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(
                    args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(
                    outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i])
                   for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)

        return loss1

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(
            ), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad *
                                                                       z + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        self.lr_scheduler.step()

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")


def create_subset(dataset, num_examples):
    indices = random.sample(range(len(dataset)), num_examples)
    return dataset.select(indices)


def filter_datasets_for_use_case(datasets, use_case):
    filtered_datasets = {}
    for key, value in datasets.items():
        if value[use_case]:
            filtered_datasets[key] = value[use_case]
    return filtered_datasets


def split_datasets(data_dict, ratio=0.7, random_state=None, shuffle=True):
    train_data = {}
    valid_data = {}
    validation_indices = {}

    for key, value in data_dict.items():
        train, valid, train_indices, valid_indices = train_test_split(
            value, range(len(value)), train_size=ratio, random_state=random_state, shuffle=shuffle)
        train_data[key] = train
        valid_data[key] = valid
        validation_indices[key] = valid_indices

    return train_data, valid_data, validation_indices


def unique_elements(lst):
    result = []
    seen = set()
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class PerplexityLoggingCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                    metrics: Dict[str, float], prefix=None, **kwargs):
        if prefix is None:
            prefix = "eval"
        eval_loss_key = f"{prefix}_loss"
        if eval_loss_key in metrics:
            loss = metrics[eval_loss_key]
            metrics[f"{prefix}_perplexity"] = math.exp(loss)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class CustomDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __getitem__(self, idx):
        return self.tensor_list[idx]

    def __len__(self):
        return len(self.tensor_list)


def get_sequences(text, tokenizer, seq_length=768, stride_ratio=0.5):
    all_token_ids = tokenizer.encode(text)

    # Generate sequences using sliding window approach
    stride_length = int(seq_length * stride_ratio)
    sequences = []
    for i in range(0, len(all_token_ids) - seq_length + 1, stride_length):
        input_ids = all_token_ids[i:i+seq_length]
        sequences.append(input_ids)

    # Truncate the last sequence if it less than seq_length
    last_sequence = sequences[-1]
    if len(last_sequence) < seq_length:
        last_sequence = last_sequence + \
            [tokenizer.pad_token_id] * (seq_length - len(last_sequence))
        sequences[-1] = last_sequence

    # Drop any remaining sequences that are less than seq_length
    sequences = [sequence for sequence in sequences if len(
        sequence) == seq_length]

    return sequences


def evaluate(model, dataloader, device, max_eval_steps):
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        # Extract input_ids and convert them to tensors
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else None

        with torch.no_grad():
            input_dict = {'input_ids': input_ids, 'labels': labels}
            outputs = model(**input_dict)

        loss = outputs.loss.repeat(input_ids.shape[0])
        losses.append(loss.detach())
        if max_eval_steps > 0 and step >= max_eval_steps:
            break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()


class CustomTrainer(Trainer):
    def __init__(self, *args, max_eval_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_perplexity = float("inf")
        self.best_model_state_dict = None
        self.no_improvement_counter = 0
        self.passed_epoch_steps = False
        self.max_eval_steps = max_eval_steps  # Add max_eval_steps as an attribute

    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, ignore_keys=None, metric_key_prefix='eval'):
        eval_loss, perplexity = evaluate(
            self.model, dataloader, self.args.device, self.max_eval_steps)

        # Check if epoch_steps are surpassed
        if self.state.epoch >= 1:
            self.passed_epoch_steps = True

        # Check for improvements if the epoch_steps are surpassed
        if self.passed_epoch_steps:
            if perplexity < self.best_perplexity:
                self.best_perplexity = perplexity
                self.best_model_state_dict = {k: v.clone().to(
                    'cpu') for k, v in self.model.state_dict().items()}
                self.no_improvement_counter = 0
            else:
                self.no_improvement_counter += 1

        # Stop training, load the best state_dict in the model, and return the best_model if the perplexity did not improve 3 times consecutively
        if self.no_improvement_counter == 3:
            if self.best_model_state_dict:
                self.model.load_state_dict(self.best_model_state_dict)
            self.model.to(self.args.device)
            self.control.should_training_stop = True
            print("Training stopped, best model loaded with Perplexity:",
                  self.best_perplexity)

        self.log({
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "epoch": self.state.epoch,
        })

        # Define num_samples as the total number of samples in the dataloader
        # num_samples = len(dataloader.dataset)

        # Initialize an instance of EvalPrediction without the 'metrics' keyword argument
        # eval_prediction = EvalPrediction(predictions=None, label_ids=None, num_samples=num_samples)
        eval_prediction = EvalPrediction(predictions=None, label_ids=None)

        # Define num_samples as the total number of samples in the dataloader
        num_samples = len(dataloader.dataset)

        # Add the num_samples attribute to the eval_prediction instance
        eval_prediction.num_samples = num_samples

        # Set the metrics dictionary
        eval_prediction.metrics = {"eval_loss": eval_loss}

        return eval_prediction

    def get_completed_steps(self):
        return self.state.global_step

def extract_prompt_response(prompt):
    human_text = None
    gpt_text = None

    for conversation in prompt['conversations']:
        if conversation['from'] == 'human':
            human_text = conversation['value']
        elif conversation['from'] == 'gpt':
            gpt_text = conversation['value']

    return human_text, gpt_text

def generate_prompt_example(context='', prompt='', response='', task='', dispreferred=False):
    final_string = ''
    
    if task:
        final_string += f"{task}"
        
    if context:
        final_string += f"Context:\n\n{context}\n\n"
    
    final_string += f"Prompt:\n\n{prompt}\n\nResponse:\n\n{response}\n\n"
    
    if dispreferred:
        final_string += "<dispreferred>"
    else:
        final_string += "<preferred>"
    
    return final_string

# Assuming a function `remove_extra_line_breaks` exists; Placeholder for now
def remove_extra_line_breaks(text):
    cleaned_text = re.sub(r'\n\n', '', text).strip()
    return cleaned_text
    
def load_or_download_dataset(pkl_path, dataset_name, splits=None):
    if os.path.exists(pkl_path):
        print(f"Loading {dataset_name} from pickle file...")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Downloading {dataset_name} dataset...")
        if splits:
            dataset = concatenate_datasets([load_dataset(dataset_name)[split] for split in splits])
        else:
            dataset = load_dataset(dataset_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

