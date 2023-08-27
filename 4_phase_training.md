#modify resources/train.json
#these are read by vars, and by extension the .py files which all import vars

#to load new variables
modify resources/train.json
warm_ratio: how far into 1st epoch it stops warming up

#warm ratio
#STRIDE_LENGTH is dynamically adjusted based on what you input, using that as a start and iterating down by divisors of 2 to find the optimal x^2 value by deriving the max modulus.
modify vars

#prep finetune data (supports Phase II, III, IV
'/home/user/env-MeZo/bin/python /home/user/TrainLLMv2/prep_data.py'

#order

#Used to learn new documents
Phase I - ./run-mezo.sh '/home/user/env-MeZo/bin/python /home/user/TrainLLMv2/mezo-pretraining.py'
* Unstructured continued pretraining on core documents (acceptable to separate by </s> token)
* 1 book

#Phase II/III are meant to be used in conjunction, but Phase II is essentially training on 'context' only
Phase II - ./run-mezo.sh '/home/user/env-MeZo/bin/python /home/user/TrainLLMv2/mezo-finetune-pretrain.py'
* Train on Context:'s (stripping 'Context:\n') (internalize context for datasets where it makes sense, such as squad, but not instruct)
* 400 squad contexts (unique)

#Phase III follows up on Phase II's context with Prompt/Response pairs
Phase III - ./run-mezo.sh '/home/user/env-MeZo/bin/python /home/user/TrainLLMv2/mezo-finetune-I-R.py'
./run-mezo.sh '/home/user/env-MeZo/bin/python /home/user/TrainLLMv2/mezo-finetune-I-R.py'
* Train on Fine-tuning examples that utilize either readily available context, or are Instruction/Response pairs with or without Context trained during pretraining (for Instruct style prompts, it makes sense to keep Context in this phase).
* 400 qa squad answers (with the preferred special token)

#Phase IV is for datasets where you are passing the model Context, Prompt, Response pairs within the prompt.
Phase IV - ./run-mezo.sh '/home/user/env-MeZo/bin/python /home/user/TrainLLMv2/mezo-finetune-C-I-R.py'
* e.g. InstructGPT

Then I'm going to see if I can ask questions of the book

I'm performing MeZO with 4-bit quantized model with a lora adapter (lora isn't necessary, but if I want to use the 4-bit quantized, I have to), which allows me somewhat of a speedup.  I've also perfected the early stopping (no more custom Trainer class, other than MeZo).  I simply handle it all within callbacks, tracking the best model based on perplexity and going with the local minimum after at least 1 epoch.

I'm ensuring I'm also not shuffling the dataset **in the trainer** (I will shuffle beforehand, but once the dataset is 'created', the model leaves it intact to preserve the stride ordering).

For now I'm using open_llama_3b because my setup is somewhat limited, but once I confirm my 'theory' works, I will expand to runpod.
Atm, the test case is to show if I can internalize Squad_v2 context and ask questions from it (I'm also setting up my own custom direct preference optimization), which hopefully will carry over to the 'core documents', it may, it may not, else I have to mix in samples, but right now, I prefer to do my use cases serially (i.e. core documents I would like to be in a serial order, with the stride sequences shuffled within the order), I'm not sure if this is necessary, but it makes things easier logically for me to understand, but I will still only have 3 files to process this (based on the strategy outlined above).
After all this, I plan on using a webui that can search the internet, supports a faiss index of the core trained documents, and supports my own hack of RLHF plus continual modifications later (but I'm told it's always pretrain then finetune, but I'm hoping I can find a decent workaround, else I have to redo finetuning each time).
I plan on extending this to quotes, lyrics, works of philosophy, logic, analogies, etc, etc.  I've been documenting what would be choice to go into such a model.

** I attempted to implement relora, but it's not 'there' yet.
