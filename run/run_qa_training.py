import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import argparse
import logging as logger

from src import *
from src.runs.qa_training import run_qa_training
from src.utils.config import Config

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for training.')
parser.add_argument('--log_to_file',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--ckpt_name',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--dataset_name',
                    default='squad',
                    type=str,
                    action='store')
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    action='store')
parser.add_argument('--max_epoches',
                    default=10,
                    type=int,
                    action='store')
parser.add_argument('--learning_rate',
                    default=1e-5,
                    type=float,
                    action='store')
parser.add_argument('--cased',
                    default=True,
                    type=int,
                    action='store')
parser.add_argument('--max_length',
                    default=512,
                    type=int,
                    action='store')
parser.add_argument('--hidden_dim',
                    default=768,
                    type=int,
                    action='store')
parser.add_argument('--lm_name',
                    default='bert-base-multilingual-cased',
                    type=str,
                    action='store')
parser.add_argument('--continue_training',
                    default=True,
                    type=int,
                    action='store')
parser.add_argument('--device',
                    default=None,
                    type=none_or_str,
                    action='store')
args = parser.parse_args()
setup_logger(logger, args.log_to_file)

config = Config(**args.__dict__)

run_qa_training(logger, config)
