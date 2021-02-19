import gc
import numpy as np
import re
import string
import torch
from torch.utils.data import DataLoader


class ExactMatch:
    def __init__(self, dataset, device):
        self.device = device
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        self.reset_state()
    
    def eval(self, model):
        self.reset_state()
        for batch_data in self.dataloader:
            item_indices = batch_data['idx']
            input_ids = batch_data['input_ids'].to(device=self.device)
            token_type_ids = batch_data['token_type_ids'].to(device=self.device)
            attention_mask = batch_data['attention_mask'].to(device=self.device)

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            y_pred_start = outputs[0].cpu().detach().numpy()
            y_pred_end = outputs[1].cpu().detach().numpy()
            y_pred = [y_pred_start, y_pred_end]

            self.add_predictions(item_indices, y_pred)
            
            # Clear some memory
            if self.device == 'cuda':
                del input_ids
                del token_type_ids
                del attention_mask
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
        score = self.result()
        return score

    def add_predictions(self, indices, y_pred):
        y_pred[0] = np.argmax(y_pred[0], axis=-1)
        y_pred[1] = np.argmax(y_pred[1], axis=-1)

        exact_matches = 0
        for (idx, y_pred_start, y_pred_end) in zip(indices, y_pred[0], y_pred[1]):
            item = self.dataset.get_full_item(idx)

            # Get prediction of answer
            y_length = sum(item['attention_mask_context'])
            if y_pred_start >= y_length or y_pred_end >= y_length or y_pred_end < y_pred_start:
                pred_answer = ''
            else:
                start_idx = item['context_offsets'][y_pred_start][0]
                end_idx = item['context_offsets'][y_pred_end][1]
                pred_answer = item['context'][start_idx:end_idx]

            # Get real answer
            if item['is_impossible']:
                real_answer = ''
            else:
                start_idx = item['context_offsets'][item['start_token_idx']][0]
                end_idx = item['context_offsets'][item['end_token_idx']][1]
                real_answer = item['context'][start_idx:end_idx]

            other_answers = item['other_answers']

            exact_matches += self._is_same_answer(real_answer, pred_answer, other_answers)
            self.predictions[idx] = pred_answer

        self.metric_score += exact_matches
        self.n_items += len(indices)

    def result(self):
        return (self.metric_score / self.n_items) if self.n_items > 0 else 0.
    
    def get_predictions(self):
        return self.predictions

    def reset_state(self):
        self.metric_score = 0.
        self.n_items = 0
        self.predictions = {}
    
    def _is_same_answer(self, real_answer, pred_answer, other_real_answers=[]):
        norm_pred_answer = self._normalize_text(pred_answer)
        if self._normalize_text(real_answer) == norm_pred_answer:
            return True
        else:
            for other_answer in other_real_answers:
                if self._normalize_text(other_answer) == norm_pred_answer:
                    return True
            return False
    
    def _normalize_text(self, text):
        text = ''.join(x for x in text if x not in set(string.punctuation))
        text = ' '.join(text.split())
        return text.lower().strip()
