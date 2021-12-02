from deep_vitabuild.utils import utils, config_utils
from deep_vitabuild.procedures import inferences_detectron, train_detectron, valid_detectron, inferences_AL

import os
import shutil
from glob import glob
import pandas as pd

MODES = ['TRAINING', 'VALIDATION', 'INFERENCE', 'ACTIVE_LEARNING', 'SEGMENTS_TRANSFER']

def main():
    gen_cfg = utils.parse_args().cfg
    gen_cfg = config_utils.get_config(gen_cfg) #Parses the cfg.yaml file in dict of dict fashion
    final_output_dir = utils.init_exp_folder(gen_cfg)
    logger, log_filename = utils.create_logger(final_output_dir, gen_cfg)

    TRAINING = eval(f'gen_cfg.{MODES[0]}.status')
    VALIDATION = eval(f'gen_cfg.{MODES[1]}.status')
    INFERENCE = eval(f'gen_cfg.{MODES[2]}.status')
    ACTIVE_LEARNING = eval(f'gen_cfg.{MODES[3]}.status')
    SEGMENTS_TRANSFER = eval(f'gen_cfg.{MODES[4]}.status')

    building_metadata, building_val_metadata = train_detectron.add_to_catalog(gen_cfg)
    dataset_dicts = train_detectron.get_building_dicts(gen_cfg.TRAINING.DATASET_DIR+'/train')
    detec_cfg = train_detectron.cfg_detectron(gen_cfg)
    
    if TRAINING:
        train_detectron.train_detectron(detec_cfg)
    
    if VALIDATION:
        detec_cfg, trainer = valid_detectron.add_val_loss(detec_cfg)
        detec_cfg = valid_detectron.inference_val(detec_cfg, gen_cfg, building_metadata)
        valid_detectron.evaluate_AP(detec_cfg, gen_cfg, trainer)

    if INFERENCE:
        if gen_cfg.INFERENCE.STRUCTURE == 'folder':
            inferences_detectron.inference_detectron_folder(detec_cfg, gen_cfg, building_metadata)
        if gen_cfg.INFERENCE.STRUCTURE == 'full':
            inferences_detectron.inference_detectron_full(detec_cfg, gen_cfg, building_metadata)
    
    if SEGMENTS_TRANSFER:
        df_segments_info = pd.read_csv(gen_cfg.SEGMENTS_TRANSFER.segments_info_path) # read the csv file
        test_segments = [item for item in gen_cfg.SEGMENTS_TRANSFER.id_segments.split(',')]
        df_test = df_segments_info[df_segments_info['idsegment'].isin(test_segments)]
        df_test.apply(lambda x: create_segment_folder(x, gen_cfg.SEGMENTS_TRANSFER.streams_dir, gen_cfg.SEGMENTS_TRANSFER.segment_image_path), axis=1)

    if ACTIVE_LEARNING:
        inferences_AL.inference_AL(detec_cfg, gen_cfg)


    #utils.count_dataset(DATASET_DIR)

if __name__ == '__main__':
    main()