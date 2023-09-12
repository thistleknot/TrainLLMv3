# TrainLLMv3
# framework for [eventually] achieving closed-book qa
#see 4-phase-training (old document)
#atm only doing phase II.  Phase I is meant to pre-train

#important files
#resources/train.json
#vars.py

#pre-training
Generate_questions.py
#writes outputted json to llm dataset format of contexts, questions, answers
prep_raw_data.py


#other files of interest

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/env/lib/python3.11/site-packages/nvidia/cusparse/lib/
export WANDB_MODE=offline
./mezo-run.sh prep_data.py
./mezo-run.sh mezo-all-phases.py

#after training
mezo-inference.py

