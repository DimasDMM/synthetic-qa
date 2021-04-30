import os
import spacy
from spacy.lang.vi import Vietnamese
from transformers import AutoTokenizer, AutoConfig

from . import *
from .. import *

def get_piece_tokenizer(artifacts_path='artifacts/', lm_name='bert-base-cased',
                        lowercase=False, max_length=512):
    save_path = get_project_path(artifacts_path, lm_name)
    tokenizer = AutoTokenizer.from_pretrained(save_path, lowercase=lowercase, use_fast=True, model_max_length=max_length,
                                              config=AutoConfig.from_pretrained(os.path.join(save_path, 'config.json')))
    return tokenizer

def get_word_tokenizer(lang_code, word2id, max_padding=512):
    if lang_code not in SPACY_DICT:
        raise Exception('There is not any Spacy dictionary defined for language "%s".' % lang_code)
    dictionary_name = SPACY_DICT[lang_code]
    nlp = Vietnamese() if dictionary_name == 'vi' else spacy.load(dictionary_name)
    return lambda context, question=None, **kwargs : lambda_word_tokenizer(
            nlp, word2id, context, question, max_padding=max_padding)

def lambda_word_tokenizer(nlp, word2id, context, question=None, max_padding=512,
                          unknown_token=UNKNOWN_TOKEN, sep_token=SEP_TOKEN, pad_token=PAD_TOKEN):
    data = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'offset_mapping': []}
    
    # Context data
    for token in nlp(context):
        token_id = word2id[token.text] if token.text in word2id else word2id[unknown_token]
        data['input_ids'].append(token_id)
        data['token_type_ids'].append(0)
        data['attention_mask'].append(1)
        data['offset_mapping'].append([token.idx, token.idx + len(token.text)])
    
    # Question data
    if question:
        data['input_ids'].append(word2id[sep_token])
        data['token_type_ids'].append(1)
        data['attention_mask'].append(0)
        data['offset_mapping'].append([0, 0])
        for token in nlp(context):
            token_id = word2id[token.text] if token.text in word2id else word2id[unknown_token]
            data['input_ids'].append(token_id)
            data['token_type_ids'].append(1)
            data['attention_mask'].append(0)
            data['offset_mapping'].append([token.idx, token.idx + len(token.text)])
    
    # Padding
    if len(data['input_ids']) < max_padding:
        for _ in range(max_padding - len(data['input_ids'])):
            data['input_ids'].append(word2id[pad_token])
            data['token_type_ids'].append(0)
            data['attention_mask'].append(0)
            data['offset_mapping'].append([0, 0])
    
    return data
