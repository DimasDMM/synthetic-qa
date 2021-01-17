#!/bin/bash

source activate tf

python ./run/run_qa_training.py \
    --ckpt_name qa_squad \
    --dataset squad \
    --max_epoches 100 \
    --batch_size 64 \
    --cased 0 \
    --lm_name bert-base-multilingual-cased \
    --hidden_dim 768 \
    --device cuda \
    --continue_training 0 \
    --log_to_file 1
