#!/bin/bash

source activate tf

python ./run/run_qa_training.py \
    --ckpt_name qa_mbert_squad \
    --dataset_name squad \
    --max_epoches 100 \
    --batch_size 32 \
    --cased 1 \
    --lm_name bert-base-multilingual-cased \
    --hidden_dim 768 \
    --device cuda \
    --continue_training 0 \
    --log_to_file logger_mbert_squad.txt
