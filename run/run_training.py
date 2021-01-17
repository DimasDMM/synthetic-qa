import argparse
import logging as logger

from src import *
from src.runs.training import run_training
from src.utils.config import Config

if __name__ != '__main__':
    exit(0)

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

parser = argparse.ArgumentParser(description='Arguments for training.')
parser.add_argument('--log_to_file',
                    default=False,
                    type=int,
                    action='store')
parser.add_argument('--ckpt_name',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--dataset',
                    default='genia',
                    type=str,
                    action='store')
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    action='store')
parser.add_argument('--max_epoches',
                    default=100,
                    type=int,
                    action='store')
parser.add_argument('--cased',
                    default=False,
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

run_training(logger, config)
