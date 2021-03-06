import numpy as np
import re

class SquadPreprocess:
    # Modifed version from https://keras.io/examples/nlp/text_extraction_with_bert/
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_tokenizer(self):
        return self.tokenizer

    def preprocess(self, idx, question, context, start_char_idx=-1, answer='', is_impossible=False):
        # Fix white spaces
        #context = re.sub(r"\s+", ' ', context).strip()
        #question = re.sub(r"\s+", ' ', question).strip()
        #answer = re.sub(r"\s+", ' ', answer).strip()

        # Encode context + question (token IDs, mask and token types)
        encoded_input = self.tokenizer(context, question, return_offsets_mapping=True, return_token_type_ids=True,
                                       padding='max_length', truncation=False, verbose=False)
        token_type_ids = encoded_input['token_type_ids']
        attention_mask = encoded_input['attention_mask']

        if is_impossible:
            # If the question is not answereable, the span is the token [CLS]
            start_token_idx = -1
            end_token_idx = -1
            context_tokens = []
            context_offsets = []
        else:
            if re.sub(r"[\s\u200b]+", '', answer).strip() == '':
                # Ignore empty answers
                return None

            # Find end token index of answer in context
            end_char_idx = start_char_idx + len(answer)
            if end_char_idx > len(context):
                raise Exception('Error parsing SQuAD item: wrong start char idx in ID "%s".' % idx)

            # Mark the character indexes in context that are in answer
            is_char_in_ans = [0] * len(context)
            for char_idx in range(start_char_idx, end_char_idx):
                is_char_in_ans[char_idx] = 1

            # Get offsets of tokens
            tokens_offsets = encoded_input['offset_mapping']
            (context_tokens, context_offsets), _ = self._get_tokens_from_offsets(
                    context, question, tokens_offsets, token_type_ids)

            # Find tokens that were created from answer characters
            ans_token_idx = []
            for idx, (start, end) in enumerate(context_offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)

            if len(ans_token_idx) == 0:
                raise Exception('Error parsing SQuAD item: answer "%s" not found in ID "%s".' % (answer, idx))

            # Find start and end token index for tokens from answer
            start_token_idx = ans_token_idx[0]
            end_token_idx = ans_token_idx[-1]

        # Create inputs
        input_ids = encoded_input['input_ids']

        # Skip if the sequence is too long
        if self.max_length < len(input_ids):
            return None
        
        # Attention mask of context
        encoded_context = self.tokenizer(context, padding=False, verbose=False)
        padding_length = self.max_length - len(encoded_context['input_ids'])
        attention_mask_context = ([1] * len(encoded_context['input_ids'])) + ([0] * padding_length)

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'attention_mask_context': attention_mask_context,
            'start_token_idx': start_token_idx,
            'end_token_idx': end_token_idx,
            'offsets': context_offsets,
            'tokens': context_tokens,
            'context': context,
            'is_impossible': is_impossible,
        }

    def _get_tokens_from_offsets(self, context, question, offsets, token_type_ids):
        context_tokens = []
        context_offsets = []
        question_tokens = []
        question_offsets = []
        last_type_idx = token_type_ids[0]
        add_to_context = True
        for (span_start, span_end), type_idx in zip(offsets, token_type_ids):
            if type_idx == last_type_idx and add_to_context:
                context_tokens.append(context[span_start:span_end])
                context_offsets.append((span_start, span_end))
            else:
                add_to_context = False
                question_tokens.append(question[span_start:span_end])
                question_offsets.append((span_start, span_end))
        return (context_tokens, context_offsets), (question_tokens, question_offsets)

    def _get_tokens_from_ids(self, input_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        tokens = [t.strip() for t in tokens]
        tokens = [re.sub(r"^\s+", '', t, flags=re.UNICODE) for t in tokens]
        tokens = [re.sub(r"^\#+", '', t) for t in tokens]
        tokens = [re.sub(r"^\_+", '', t) for t in tokens]
        tokens = [re.sub(r"^\▁+", '', t) for t in tokens]
        return tokens
