import json
import os
from .. import *
from .squad import *

def build_squad_dataset(dataset_name, tokenizer, max_length):
    file_path = get_dataset_path(dataset_name)
    with open(file_path) as fp:
        raw_data = json.load(fp)
    squad_examples = build_squad_objects(raw_data, tokenizer, max_length)
    return squad_examples

def get_dataset_path(dataset_name):
    if dataset_name == 'squad-train':
        return get_project_path('data/squad/train-v1.1.json')
    elif dataset_name == 'squad-test':
        return get_project_path('data/squad/dev-v1.1.json')
    elif dataset_name == 'synthetic-train':
        return get_project_path('data/synthetic_v1/mkqa_squad.json')

def build_squad_objects(raw_data, tokenizer, max_length):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                start_char_idx = qa["answers"][0]["answer_start"]
                squad_eg = SquadExample(
                    question,
                    context,
                    start_char_idx,
                    answer_text,
                    all_answers,
                    tokenizer,
                    max_length
                )
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples

def split_inputs_targets(squad_examples):
    dataset_dict = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'start_token_idx': [],
        'end_token_idx': [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = {
        'input_ids': dataset_dict['input_ids'],
        'token_type_ids': dataset_dict['token_type_ids'],
        'attention_mask': dataset_dict['attention_mask'],
    }
    y = {'start_logits': dataset_dict['start_token_idx'], 'end_logits': dataset_dict['end_token_idx']}
    return x, y
