import logging
import shutil

import yaml
from easydict import EasyDict as edict

from pathlib import Path
import argparse

def init_exp_folder(cfg):
    if 'FAKERUN' in cfg and cfg.FAKERUN:
        return None

    output_dir = cfg.OUTPUT_DIR
    exp_foldername = cfg.EXP_NAME

    assert exp_foldername is not None, 'Specify the Experiment Name!'
    assert output_dir is not None, 'Specify the Output folder (where to save models, checkpoints, etc.)!'
    root_output_dir = Path(output_dir)

    # set up logger
    if not root_output_dir.exists():
        print(f'=> creating "{root_output_dir}" ...')
        root_output_dir.mkdir()
    else:
        print(f'Folder "{root_output_dir}" already exists.')

    final_output_dir = root_output_dir / exp_foldername

    if not final_output_dir.exists():
        print('=> creating "{}" ...'.format(final_output_dir))
    else:
        print(f'Folder "{final_output_dir}" already exists.')

    final_output_dir.mkdir(parents=True, exist_ok=True)
    return final_output_dir

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

def copy_exp_file(trainer):
    if 'FAKERUN' in trainer.cfg and trainer.cfg.FAKERUN:
        return
    shutil.copy2(trainer.cfg.CONFIG_FILENAME, trainer.final_output_dir)

def copy_proc_file(trainer):
    if 'FAKERUN' in trainer.cfg and trainer.cfg.FAKERUN:
        return
    
    # TODO: fix path
    proc_file = f'./src/deep_cvlab/procedures/procedures/{trainer.cfg.PROCEDURE}.py'
    #shutil.copy2(proc_file, trainer.final_output_dir)
    print("No preocedure system implemented")