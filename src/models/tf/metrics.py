import numpy as np
import re
import string
import tensorflow as tf

# Modified version of: https://keras.io/examples/nlp/text_extraction_with_bert/
class ExactMatch(tf.keras.metrics.Metric):
    def __init__(self, X_data, y_data, squad_examples):
        self.X_data = X_data
        self.y_data = y_data
        self.squad_examples = squad_examples

        self.metric_score = self.add_weight(name='metric_score', initializer='zeros')
        self.n_items = self.add_weight(name='n_items', initializer='zeros')
        self.update_metric = tf.Variable(False)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self.update_metric:
            return

        n_items = len(y_true)
        pred_start, pred_end = y_pred
        count = 0
        eval_examples_no_skip = [_ for _ in self.squad_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            
            # Get answer from context text
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            # Normalize answers before comparing prediction and true answers
            normalized_pred_ans = self._normalize_text(pred_ans)
            normalized_true_ans = [self._normalize_text(_) for _ in squad_eg.all_answers]
            
            # If the prediction is contained in the true answer, it counts as a hit
            if normalized_pred_ans in normalized_true_ans:
                count += 1

        score = count / len(self.y_data[0])
        self.metric_score.assign_add(score)
        self.n_items.assign_add(n_items)

    def result(self):
        if self.n_items == 0.:
            return self.n_items
        else:
            return tf.divide(self.metric_score, self.n_items)

    def reset_states(self):
        self.metric_score.assign(0.)
        self.n_items.assign(0.)
    
    def _normalize_text(self, text):
        text = text.lower()

        # Remove punctuations
        exclude = set(string.punctuation)
        text = ''.join(ch for ch in text if ch not in exclude)

        # Remove extra white spaces
        text = re.sub(r"\s+", ' ', text).strip()

        return text
