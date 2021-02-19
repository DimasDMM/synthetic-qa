import json
import os
import torch
from .. import *
from .squad import *


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, squad_preprocess, dataset_path, save_contexts=True, logger=None):
        # Non-tensor attributes
        self.logger = logger
        self.skipped_idx = []
        self.n_items = 0
        self.save_contexts = save_contexts
        
        self.idx = []
        self.context_tokens = []
        self.context_offsets = []
        self.contexts = []
        self.answerables = []
        self.other_answers = []

        # Tensor attributes
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.attention_mask_context = []
        self.start_token_idx = []
        self.end_token_idx = []

        with open(dataset_path) as fp:
            raw_data = json.load(fp)
        self._build_squad_objects(raw_data, squad_preprocess)

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        self.attention_mask_context = torch.tensor(self.attention_mask_context, dtype=torch.long)
        self.start_token_idx = torch.tensor(self.start_token_idx, dtype=torch.long)
        self.end_token_idx = torch.tensor(self.end_token_idx, dtype=torch.long)

        self.idx2pos = {idx:i for i, idx in enumerate(self.idx)}

    def get_item(self, idx):
        i = self.idx2pos[idx]
        return self.__getitem__(i)

    def get_full_item(self, idx):
        i = self.idx2pos[idx]
        item = self.__getitem__(i)
        return {
            'idx': item['idx'],
            'context_tokens': self.context_tokens[i],
            'context_offsets': self.context_offsets[i],
            'input_ids': item['input_ids'],
            'token_type_ids': item['token_type_ids'],
            'attention_mask': item['attention_mask'],
            'attention_mask_context': item['attention_mask_context'],
            'start_token_idx': item['start_token_idx'],
            'end_token_idx': item['end_token_idx'],
            'context': self.contexts[i],
            'is_impossible': self.answerables[i],
            'other_answers': self.other_answers[i],
        }
    
    def get_skipped_items(self):
        return self.skipped_idx

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        return {
            'idx': self.idx[i],
            'input_ids': self.input_ids[i],
            'token_type_ids': self.token_type_ids[i],
            'attention_mask': self.attention_mask[i],
            'attention_mask_context': self.attention_mask_context[i],
            'start_token_idx': self.start_token_idx[i],
            'end_token_idx': self.end_token_idx[i],
        }

    def _build_squad_objects(self, raw_data, squad_preprocess):
        n_items = len(raw_data['data'])
        for i, item in enumerate(raw_data['data']):
            if i % 50 == 0 and self.logger is not None:
                self.logger.info('- Processed %d / %d...' % (i + 1, n_items))
            for para in item['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    idx = qa['id']
                    question = qa['question']
                    is_impossible = ('is_impossible' in qa and qa['is_impossible'])
                    if len(qa['answers']) == 0:
                        squad_item = squad_preprocess.preprocess(idx, question, context, is_impossible=is_impossible)
                    else:
                        answer_text = qa['answers'][0]['text']
                        other_answers = [x['text'] for x in qa['answers'][1:]]
                        start_char_idx = qa['answers'][0]['answer_start']
                        squad_item = squad_preprocess.preprocess(idx, question, context, start_char_idx,
                                answer_text, is_impossible=is_impossible)
                    if squad_item is None:
                        self.skipped_idx.append(idx)
                        continue
                    self.n_items += 1
                    self._add_item(idx, squad_item, other_answers)
    
    def _add_item(self, idx, squad_item, other_answers=[]):
        self.idx.append(idx)
        self.context_tokens.append(squad_item['tokens'])
        self.context_offsets.append(squad_item['offsets'])
        self.input_ids.append(squad_item['input_ids'])
        self.token_type_ids.append(squad_item['token_type_ids'])
        self.attention_mask.append(squad_item['attention_mask'])
        self.attention_mask_context.append(squad_item['attention_mask_context'])
        self.start_token_idx.append(squad_item['start_token_idx'])
        self.end_token_idx.append(squad_item['end_token_idx'])
        self.answerables.append(squad_item['is_impossible'])
        self.other_answers.append(other_answers)
        if self.save_contexts:
            self.contexts.append(squad_item['context'])
        else:
            self.contexts.append('')
