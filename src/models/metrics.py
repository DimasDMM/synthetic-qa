import collections
import gc
import numpy as np
import re
import string
import torch
from torch.utils.data import DataLoader


class ModelPredictions:
    def __init__(self, device):
        self.device = device
    
    def get_predictions(self, model, dataloader, dataset):
        predictions = {}
        for batch_data in dataloader:
            item_indices = batch_data['idx']
            input_ids = batch_data['input_ids'].to(device=self.device)
            token_type_ids = batch_data['token_type_ids'].to(device=self.device)
            attention_mask = batch_data['attention_mask'].to(device=self.device)

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            y_pred_start = outputs[0].cpu().detach().numpy()
            y_pred_end = outputs[1].cpu().detach().numpy()
            y_pred = [y_pred_start, y_pred_end]

            self._add_predictions(item_indices, y_pred, dataset, predictions)
            
            # Clear some memory
            if self.device == 'cuda':
                del input_ids
                del token_type_ids
                del attention_mask
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
        
        return predictions

    def _add_predictions(self, indices, y_pred, dataset, predictions={}):
        y_pred[0] = np.argmax(y_pred[0], axis=-1)
        y_pred[1] = np.argmax(y_pred[1], axis=-1)

        skipped_items = 0
        for (idx, y_pred_start, y_pred_end) in zip(indices, y_pred[0], y_pred[1]):
            item = dataset.get_full_item(idx)

            # Unanswereable item
            if item['start_token_idx'] < 0:
                skipped_items += 1
                continue

            # Get prediction of answer
            y_length = sum(item['attention_mask_context'])
            if y_pred_start >= y_length or y_pred_end >= y_length or y_pred_end < y_pred_start:
                pred_answer = ''
            else:
                start_idx = item['context_offsets'][y_pred_start][0]
                end_idx = item['context_offsets'][y_pred_end][1]
                pred_answer = item['context'][start_idx:end_idx]

            predictions[idx] = pred_answer
        
        return predictions


class BaseMetric:
    def __init__(self):
        pass
    
    def eval(self, dataset, predictions):
        metric_score = 0
        skipped_items = 0

        for pred_idx, pred_answer in predictions.items():
            real_item = dataset.get_full_item(pred_idx)

            # Unanswereable item
            if real_item['start_token_idx'] < 0:
                skipped_items += 1
                continue
            
            # Get all possible real answers
            all_real_answers = real_item['all_answers']

            metric_score += self._compute_item_score(all_real_answers, pred_answer)

        n_items = len(predictions) - skipped_items
        score = (metric_score / n_items) if n_items > 0 else 0.
        return score, n_items

    def _compute_item_score(self, all_real_answers, pred_answer):
        raise NotImplementedError()


class ExactMatchMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def _compute_item_score(self, all_real_answers, pred_answer):
        norm_pred_answer = self._normalize_text(pred_answer)
        for real_answer in all_real_answers:
            if self._normalize_text(real_answer) == norm_pred_answer:
                return True
        return False
    
    def _normalize_text(self, text):
        text = ''.join(x for x in text if x not in set(string.punctuation))
        text = ' '.join(text.split())
        return text.lower().strip()


class F1ScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def _compute_item_score(self, all_real_answers, pred_answer):
        all_gold_toks = [self._get_tokens(x) for x in all_real_answers]
        pred_toks = self._get_tokens(pred_answer)
        f1_score = max([self._f1_score(gold_toks, pred_toks) for gold_toks in all_gold_toks])
        return f1_score
    
    def _f1_score(self, gold_toks, pred_toks):
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def _normalize_text(self, text):
        text = ''.join(x for x in text if x not in set(string.punctuation))
        text = ' '.join(text.split())
        return text.lower().strip()
    
    def _get_tokens(self, s):
        if not s:
            return []
        return self._normalize_text(s).split()
