import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import argparse
import logging as logger
import os
from transformers import AutoModel, AutoTokenizer
from src import *

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for LM Download.')
parser.add_argument('--log_to_file',
                    default=None,
                    type=none_or_str,
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
slow_tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
slow_tokenizer.save_pretrained(save_path)

logger.info('Download model')
model = AutoModel.from_pretrained(args.lm_name)

model.save_pretrained(save_path)

logger.info('Done')
