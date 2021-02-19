# Synthetic-QA

## Introduction

TODO

## Set up

TODO

Prepare pretrained BERT model:
```sh
python ./run/run_download_lm.py --lm_name bert-base-multilingual-cased
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
