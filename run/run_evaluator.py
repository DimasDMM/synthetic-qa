import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import argparse
import logging as logger

from src import *
from src.runs.evaluator import run_evaluator
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

parser = argparse.ArgumentParser(description='Arguments for evaluation.')
parser.add_argument('--log_to_file',
                    default=False,
                    type=int,
                    action='store')
parser.add_argument('--model_ckpt',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--dataset',
                    default='genia',
                    type=str,
                    action='store')
parser.add_argument('--device',
                    default=None,
                    type=none_or_str,
                    action='store')
args = parser.parse_args()
setup_logger(logger, args.log_to_file)

config = Config(**args.__dict__)

run_evaluator(logger, config)
