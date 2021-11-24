from deep_vitabuild.utils import utils, config_utils
from deep_vitabuild import core
from deep_vitabuild.procedures import inferences_detectron, train_detectron, valid_detectron

import os
import shutil
import argparse
from glob import glob
import pandas as pd

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_segment_folder(df, streams_dir, segment_dir):
    id_segment = df['idsegment']
    # find the streams taken by panorana camera (the second item of streams)
    panorama_stream = df['streams'].split(',')[1]

    make_dir(f'{segment_dir}/{id_segment}')
    former_address = f'{streams_dir}/{panorama_stream}'
    frames = sorted(glob(f'{former_address}/*'))
    for frame in frames:
        frame_num = frame.split('/')[-1]
        # only use the direction 1 and direction 4 (left and right side of panorama camera)
        imgs = glob(frame+'/[14].jpg')
        for img in imgs:
            img_name = img.split('/')[-1]
            print(f'{segment_dir}/{id_segment}/{id_segment}_{panorama_stream}_{frame_num}_{img_name}')
            shutil.copy(
                img, f'{segment_dir}/{id_segment}/{id_segment}_{panorama_stream}_{frame_num}_{img_name}')

def test_folder():
    DATASET_DIR = '/data/facades/dataset/annotated_data/build_seg_annotation1/build_seg_annotations1'
    TARGET_PATH = '/data/facades/dataset/annotated_data/build_seg_annotation1/basel_annotations1'

    os.makedirs(TARGET_PATH, exist_ok=True)
    
    print(next(os.walk(DATASET_DIR))[1])
    print('hello')
    for folder in next(os.walk(DATASET_DIR))[1]:
        folder_path = DATASET_DIR+'/'+folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                print(filename)
                shutil.copy(
                f'{folder_path}/{filename}', f'{TARGET_PATH}/{filename}')

    
    return 0

def main():
    TRAIN = 1
    VALID = 1
    INFERENCES = 0
    SEGMENTS_TRANSFER = 0

    gen_cfg = utils.parse_args().cfg
    gen_cfg = config_utils.get_config(gen_cfg) #Parses the cfg.yaml file in dict of dict fashion
    final_output_dir = utils.init_exp_folder(gen_cfg)
    logger, log_filename = utils.create_logger(final_output_dir, gen_cfg)

    print(tuple(gen_cfg.DETECTRON.CATALOG))
    print(gen_cfg.TRAINING)

    building_metadata, building_val_metadata = train_detectron.add_to_catalog(gen_cfg)
    dataset_dicts = train_detectron.get_building_dicts(gen_cfg.TRAINING.DATASET_DIR+'/train')
    detec_cfg = train_detectron.cfg_detectron(gen_cfg)

    print(detec_cfg.OUTPUT_DIR)
    
    if TRAIN:
        train_detectron.train_detectron(detec_cfg)
    
    if VALID:
        detec_cfg, trainer = valid_detectron.add_val_loss(detec_cfg)
        detec_cfg = valid_detectron.inference_val(detec_cfg, gen_cfg, building_metadata)
        valid_detectron.evaluate_AP(detec_cfg, gen_cfg, trainer)

    if INFERENCES:
        if gen_cfg.INFERENCE.STRUCTURE == 'folder':
            inferences_detectron.inference_detectron_folder(detec_cfg, gen_cfg, building_metadata)
        if gen_cfg.INFERENCE.STRUCTURE == 'full':
            inferences_detectron.inference_detectron_full(detec_cfg, gen_cfg, building_metadata)
    
    if SEGMENTS_TRANSFER:
        df_segments_info = pd.read_csv(gen_cfg.SEGMENTS_TRANSFER.segments_info_path) # read the csv file
        test_segments = [item for item in gen_cfg.SEGMENTS_TRANSFER.id_segments.split(',')]
        df_test = df_segments_info[df_segments_info['idsegment'].isin(test_segments)]
        df_test.apply(lambda x: create_segment_folder(x, gen_cfg.SEGMENTS_TRANSFER.streams_dir, gen_cfg.SEGMENTS_TRANSFER.segment_image_path), axis=1)


    #utils.count_dataset(DATASET_DIR)
    test_folder()

if __name__ == '__main__':
    main()