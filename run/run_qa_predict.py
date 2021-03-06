import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import argparse
import logging as logger

from src import *
from src.runs.qa_predict import run_qa_predict
from src.utils.config import Config

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for evaluation.')
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
parser.add_argument('--device',
                    default=None,
                    type=none_or_str,
                    action='store')
args = parser.parse_args()
setup_logger(logger, args.log_to_file)

config = Config(**args.__dict__)

run_qa_predict(logger, config)
