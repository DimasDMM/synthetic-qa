# Synthetic-QA

## Introduction

QA models performance is conditioned to the dataset quality and quantity.
However, it is not always easy to find this data when it comes to
annotated datasets since it requires a lot of human effort to build it. Aside
from this problem, it is even more difficult to obtain either monolingual
or cross-lingual annotated data for Natural Language Processing tasks in
other languages than English.

This project aims to define a framework to obtain an annotated dataset in any
language to train a model for reading comprehension and question-answering
tasks by using online tools like Wikipedia and Google.

## Set up

Clone this repository, create default folders and install the Python
dependencies:
```sh
git clone git@github.com:DimasDMM/synthetic-qa.git
cd synthetic-qa
mkdir data
mkdir artifacts
pip install -r requirements.txt
```

It is also recommended to download a pre-trained BERT model beforehand
(especially if we are going to use some computer without internet access):
```sh
python ./run/run_download_lm.py --lm_name bert-base-multilingual-cased
```

Optionally, if you are willing to run the sample commands, you will need to
download the SQuAD dataset:
```sh
cd data
mkdir squad
cd squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json --no-check-certificate
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json --no-check-certificate
cd ./../..
```

## Commands

### Training

Command example:
```sh
python ./run/run_qa_training.py \
    --ckpt_name qa_mbert_squad \
    --dataset_train_path ./data/squad/train-v1.1.json \
    --dataset_dev_path ./data/squad/dev-v2.0.json \
    --max_epoches 100 \
    --batch_size 32 \
    --cased 1 \
    --lm_name bert-base-multilingual-cased \
    --hidden_dim 768 \
    --device cuda \
    --continue_training 0 \
    --log_to_file logger_mbert_squad.txt
```

Available options:
- `log_to_file` (optional, default: nothing): file to store the log output of
  the training script.
- `model_type` (optional, default: "transformers"): this option is only
  **experimental** since we may add more type of models in the future.
- `ckpt_name` (optional, default: "model_ckpt"): subfolder of `./artifacts` to
  store the model with the best weights.
- `dataset_train_path`: file with the train dataset.
- `dataset_dev_path`: file with the validation dataset.
- `batch_size` (optional, default: 32): batch size for training.
- `max_epoches` (optional, default: 10): maximum number of epoches for training.
- `learning_rate` (optional, default: 1e-5): learning rate of Adam optimizer.
- `cased` (optional, default: 1): denotes if the model should handle cased
  tokens (default behaviour), or should lowercase all tokens (when is `0`).
- `max_length` (optional, default: 512): maximum number of tokens that the model
  is able to handle.
- `hidden_dim` (optional, default: 768): output dimension of transformer model.
- `lm_name` (optional, default: "bert-base-multilingual-cased"): name of the
  pretrained model we want to use.
- `continue_training` (optional, default: 0): flag to resume the training of a
  model. Note that if the model already exists in the folder `./artifacts` and
  this flag is set to `0`, the script will output an error.
- `device` (optional, default: nothing): device to use for training: `cuda` or
  `cpu`. If nothing is provided, it will let Pytorch decide which device should
  use.

### Predictions

Command example:
```sh
python ./run/run_qa_predict.py \
    --ckpt_name qa_mbert_squad \
    --dataset_test_path ./data/squad/dev-v1.1.json \
    --output_pred_file dev-v1.1.json \
    --device cuda
```

Available options:
- `log_to_file` (optional, default: nothing): file to store the log output of
  the prediction script.
- `model_type` (optional, default: "transformers"): this option is only
  **experimental** since we may add more type of models in the future.
- `ckpt_name` (optional, default: "model_ckpt"): subfolder of `./artifacts` to
  load the model to make predictions.
- `dataset_test_path`: file with the test dataset.
- `output_pred_path` (optional, default: "artifacts/predictions"): path where
  the predictions will be stored.
- `output_pred_file`: file where the predictions will be stored.
- `device` (optional, default: nothing): device to use for training: `cuda` or
  `cpu`. If nothing is provided, it will let Pytorch decide which device should
  use.

### Evaluation

> This evaluation script is strongly based on the evaluation script provided in
> the [SQuAD website](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).

Command example:
```sh
python ./run/run_qa_evaluate.py \
    --data_file ./data/squad/dev-v1.1.json \
    --pred_file ./artifacts/predictions_normal/qa_mbert_squad/dev-v1.1.json \
    --out-file ./artifacts/predictions_normal/qa_mbert_squad/dev-v1.1_metrics.json
```

Available options:
- `data_file`: file that we used to make the predictions.
- `pred_file`: file where the predictions are stored.
- `out_file`: file where the evaluation metrics will be stored.

---

Have fun! ᕙ (° ~ ° ~)
