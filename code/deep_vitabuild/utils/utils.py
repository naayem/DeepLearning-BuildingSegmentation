import logging
import shutil

import yaml
from easydict import EasyDict as edict

from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Launch')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/exp.yaml',
                        required=False,
                        type=str)
    args, rest = parser.parse_known_args()
    args = parser.parse_args()
    return args