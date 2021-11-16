# install detectron2: (Colab has CUDA 10.2 + torch 1.8)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
import warnings
import gdown
import shutil
import random
import cv2
import json
import os
from helper import *
from tqdm import tqdm
from glob import glob
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import argparse
import pandas as pd
import numpy as np
from detectron2.utils.logger import setup_logger
import detectron2
import torch
# need to manually install torch 1.8 if Colab changes its default version
assert torch.__version__.startswith("1.8")
setup_logger()

warnings.filterwarnings("ignore")


def model2(dataset_dir, predicted_img_dir, building_info_dir):

    make_dir(predicted_img_dir)
    make_dir(building_info_dir)
    dataset = sorted(glob(os.path.join(dataset_dir, "*.jpg")))

    print('============  running model2 ============ ')
    for file_address in tqdm(dataset):

        # building location infos
        filename = file_address.split('/')[-1]
        filename_0 = filename.split('.')[0]
        info = filename.split('_')
        id_segment, id_stream, id_frame = info[0], info[1], info[2]
        # if len(id_frame) ==1:
        #     id_frame = f'0{id_frame}'
        id_direction = info[-1].split('.')[0]

        segment_dir = f'{predicted_img_dir}/{id_segment}'
        segment_opening_dir = f'{building_info_dir}/{id_segment}'
        make_dir(segment_dir)
        make_dir(segment_opening_dir)

        # use mask rcnn model to get prediction
        im = cv2.imread(file_address)
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=building_metadata,
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # instance, mask, classes
        origin_image = im
        output_image = out.get_image()[:, :, ::-1]

        # save predicted image
        savepath = f'{segment_dir}/{id_segment}_{id_stream}_{id_frame}_{id_direction}.jpg'
        cv2.imwrite(savepath, output_image)

        ins = outputs["instances"].to("cpu")
        masks = ins.get_fields()["pred_masks"].numpy()
        classes = ins.get_fields()["pred_classes"].numpy()

        indexs_masonry = np.where(classes == 0)[0]
        indexs_m6 = np.where(classes == 1)[0]
        indexs_rcw = np.where(classes == 2)[0]

        building_type_name = ['masonry', 'm6', 'rcw']
        num_building = len(classes)

        if num_building == 0:
            df_building = pd.DataFrame({'file_name': [filename]})
            df_building[['building_type',
                         'building_centroid_coordinate']] = np.nan, np.nan
        else:
            # building type
            df_building = pd.DataFrame(
                {'building_type': [building_type_name[i] for i in classes]})
            # find builidng centroid coordinate
            df_building['building_centroid_coordinate'] = find_building_centroid(
                origin_image, masks)
            df_building['file_name'] = filename
            df_building = df_building.iloc[:, [-1, 0, 1]]

        # save dataframe
        df_building.to_csv(f'{segment_opening_dir}/{filename_0}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--weight_path', default='./data/model_weight/model2_weight.pth',
                        type=str, help='the trained weight of model2')
    parser.add_argument('--data_path', default='/path/to/data',
                        type=str, help="data path of train and val set of model2")
    parser.add_argument('--segment_image_path', default='/path/to/segments', type=str,
                        help="data folder to save the building images from the panorama camera. Images are saved in segment folders")
    parser.add_argument('--id_segments', default='16888', type=str,
                        help="segments' id for testing(delimited list input)")
    parser.add_argument('--result_path', default='/path/to/result',
                        type=str, help="result folder of building type and centroid ")
    args = parser.parse_args()

    print('============  Arguments infos ============ ')
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))

    # ============ preparing model  ============
    SEGMENTS_FOLDER = args.segment_image_path
    RESULT_FOLDER = args.result_path
    DATA_FOLDER = args.data_path
    DATASET_ADDRESS = f'{DATA_FOLDER}/dataset'
    # make_dir(f'./result')
    make_dir(f'{RESULT_FOLDER}')
    # make_dir(f'{RESULT_FOLDER}/weight')

    # Load model weight
    weight_address = args.weight_path
    url = 'https://drive.google.com/uc?id=1cVdWLB2G-2CfVQS6mNqoB0d20AE8Lkm3'
    if not os.path.exists(weight_address):
        gdown.download(url, weight_address, quiet=False)

    # add the dict to Cataclog

    # set four categories, opening and 3 building type, give their id '0', '1', '2', '3'.
    dict_category_id = {
        'masonry': 0,
        'm6': 1,
        'rcw': 2
    }
    for d in ["train", "val"]:
        DatasetCatalog.register(
            'building_'+d, lambda d=d: get_building_dicts(DATASET_ADDRESS+'/' + d, dict_category_id))
        MetadataCatalog.get(
            'building_'+d).set(thing_classes=["masonry", "m6", "rcw"])
    building_metadata = MetadataCatalog.get("building_train")

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:

    # === changed parameters ===
    EPOCH = 20
    TOTAL_NUM_IMAGES = 449
    # ===========================

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("building_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2   # changed parameters
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = int(
        EPOCH*TOTAL_NUM_IMAGES/cfg.SOLVER.IMS_PER_BATCH-1)
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # changed parameters
    # 3 class ("masonry","m6","rcw"). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3     # changed parameters

    # weights
    cfg.MODEL.WEIGHTS = weight_address  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80   # set a custom testing threshold

    # prediction
    predictor = DefaultPredictor(cfg)

    # ============ run model2 to predict building type  ============
    predicted_img_dir = f'{RESULT_FOLDER}/predicted_imgs'
    building_info_dir = f'{RESULT_FOLDER}/building_info'
    test_segments = [item for item in args.id_segments.split(',')]

    for id_segment in test_segments:
        dataset_dir = f'{SEGMENTS_FOLDER}/{str(id_segment)}'
        model2(dataset_dir, predicted_img_dir, building_info_dir)
        merge_opening_info_csv(
            path=f'{building_info_dir}/{id_segment}', id_segment=id_segment)

    print(f'============ Finish, result is saved at {args.result_path}  ============ ')
