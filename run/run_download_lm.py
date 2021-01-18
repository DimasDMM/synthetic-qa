import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import argparse
import logging as logger
import os
from transformers import BertModel, TFBertModel
from transformers import BertTokenizer
from src import *

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for LM Download.')
parser.add_argument('--log_to_file',
                    default=False,
                    type=int,
                    action='store')
parser.add_argument('--lm_name',
                    default='bert-base-multilingual-cased',
                    type=str,
                    action='store')
parser.add_argument('--model_type',
                    default='tf',
                    type=str,
                    action='store')
args = parser.parse_args()
setup_logger(logger, args.log_to_file)

logger.info('== LM DOWNLOADER ==')

artifacts_path = get_project_path('artifacts')
save_path = os.path.join(artifacts_path, args.lm_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger.info('Download tokenizer')
slow_tokenizer = BertTokenizer.from_pretrained(args.lm_name)
slow_tokenizer.save_pretrained(save_path)

logger.info('Download model')
if model_type == 'tf':
    model = TFBertModel.from_pretrained(args.lm_name)
elif model_type == 'torch':
    model = BertModel.from_pretrained(args.lm_name)
else:
    raise Exception('Unknown value for model_type: %s' % model_type)

model.save_pretrained(save_path)

logger.info('Done')
