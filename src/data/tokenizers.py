import os
import re
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def get_tokenizer(artifacts_path='./artifacts/', lm_name='bert-base-multilingual-cased', lowercase=True):
    save_path = '%s%s/' % (artifacts_path, lm_name)
    tokenizer = BertWordPieceTokenizer('%svocab.txt' % save_path, lowercase=lowercase)
    return tokenizer
