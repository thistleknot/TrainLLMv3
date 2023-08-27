#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/env/lib/python3.11/site-packages/nvidia/cusparse/lib/
export WANDB_MODE=offline

/home/user/env-MeZo/bin/python ./$1
#/home/user/env-MeZo/bin/python ./mezo-inference.py
