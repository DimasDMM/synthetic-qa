#!/bin/bash

source activate tf

python ./run/run_training.py \
    --model_ckpt ./artifacts/qa_squad/ \
    --dataset squad \
    --max_epoches 100 \
    --total_layers 16 \
    --batch_size 64 \
    --cased 0 \
    --hidden_dim 100 \
    --lm_name bert-base-multilingual-cased \
    --lm_emb_dim 768 \
    --device cuda \
    --continue_training 0 \
    --log_to_file 1
