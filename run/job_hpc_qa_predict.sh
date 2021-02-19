#!/bin/bash

source activate tf

python ./run/run_qa_predict.py \
    --ckpt_name qa_squad2 \
    --dataset_name squad2 \
    --device cuda
