# Synthetic-QA

## Introduction

TODO

## Set up

TODO

Prepare pretrained BERT model:
```sh
python ./run/run_download_lm.py --lm_name bert-base-multilingual-cased
```

Run QA training:
```sh
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
```

## Commands

TODO

Evaluate predictions:
```sh
python ./run/run_qa_evaluate.py \
    --data_file ./data/squad/dev-v1.1.json \
    --pred_file ./artifacts/qa_squad/predictions/squad.json \
    --out-file ./artifacts/qa_squad/predictions/squad_metrics.json \
    --out-image-dir ./artifacts/qa_squad/predictions/metrics.jpg
```
