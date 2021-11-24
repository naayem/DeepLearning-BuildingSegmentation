# install detectron2: (Colab has CUDA 10.2 + torch 1.8)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
#from _typeshed import Self
import warnings, gdown, shutil, random, cv2, json, os
import torch, torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
#from helper import *
#from floor_helper import *

# need to manually install torch 1.8 if Colab changes its default version
assert torch.__version__.startswith("1.8")
setup_logger()

warnings.filterwarnings("ignore")


def model1(dataset_dir, cropped_img_dir, opening_info_dir, floor_dir):

    make_dir(cropped_img_dir)
    make_dir(opening_info_dir)
    make_dir(floor_dir)
    dataset = sorted(glob(os.path.join(dataset_dir, "*.jpg")))

    print('============  running model1 ============ ')
    for file_address in tqdm(dataset):

        # building location infos
        filename = file_address.split('/')[-1]
        filename_0 = filename.split('.')[0]
        info = filename.split('_')
        id_segment, id_stream, id_frame = info[0], info[1], info[2]
        # if len(id_frame) ==1:
        #   id_frame = f'0{id_frame}'
        id_direction = info[-1].split('.')[0]

        segment_dir = f'{cropped_img_dir}/{id_segment}'
        segment_opening_dir = f'{opening_info_dir}/{id_segment}'
        segment_floor_dir = f'{floor_dir}/{id_segment}'
        make_dir(segment_dir)
        make_dir(segment_opening_dir)
        make_dir(segment_floor_dir)

        # use mask rcnn model to get prediction
        im = cv2.imread(file_address)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                   metadata=building_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
         )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = out.get_image()[:, :, ::-1]


        # building masks, bounding boxes, and category classes of buildings
        ins = outputs["instances"].to("cpu")
        masks = ins.get_fields()["pred_masks"].numpy()
        classes = ins.get_fields()["pred_classes"].numpy()
        boxes = ins.get_fields()["pred_boxes"].tensor.cpu().numpy()
        indexs_building = np.where(classes == 1)[0]

        # ============ crop the building instances  ============
        for idx, index_building in enumerate(indexs_building):

            box = boxes[index_building]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # **** crop building instnce whose width >= 500****
            if (abs(x2-x1) >= 500):
                # crop the mask and fill white pixel outside the masks
                img_mask = masks[index_building]*255
                img_mask = img_mask.astype('uint8')
                img_input = im.copy()
                masked = cv2.bitwise_and(img_input, img_input, mask=img_mask)
                mask_output = np.where(masked == 0, 255, masked)
                cropped_img = mask_output[y1:y2, x1:x2]
                # save cropped imgs
                savepath = f'{segment_dir}/{id_segment}_{id_stream}_{id_frame}_{id_direction}_{str(index_building)}.jpg'
                cv2.imwrite(savepath, cropped_img)

        # ============ opening infos ============
        indexs_opening = np.where(classes == 0)[0]
        if len(indexs_building) == 0:
            df_building = pd.DataFrame({'file_name': [filename]})
            df_building[['building_index', 'building_centroid_coordinate',
                         'opening_index_inside', 'opening_facade_ratio']] = np.nan, np.nan, np.nan, np.nan
        else:
            df_building = pd.DataFrame({'building_index': indexs_building})
            # find opening inside building
            find_opening_insidie_building(
                im, masks, indexs_building, indexs_opening, df_building)
            # get opening/facade ratio
            get_opening_facade_ratio(
                im, masks, indexs_building, indexs_opening, df_building)

            df_building['file_name'] = filename
            df_building = df_building.iloc[:, [-1, 0, 1, 2, 3]]


        # ============ draw the buildng floors  ============
        range_threshold = 0.60
        r2_socre_threshold = 0.0
        delta_x_threshold = 300

        # get centroid for each mask
        dict_centroid = get_centroid(im, masks)
        # draw label to the centroid of mask
        predicted_image = process_output(im, output_image, dict_centroid)
        
        list_floor_info, list_num_floor = [], [] 
        for i in df_building.building_index:
          if not np.isnan(i):
            floor_img =  predicted_image
            slope_threshold_1, slope_threshold_2 = handle_slope_threshold(im, i, masks, boxes, range_threshold, r2_socre_threshold, degree_close =25, degree_far=15, flag=True)
            floor_img, list_floor_centroid, num_floor = run_plot_whole(floor_img, df_building, dict_centroid, boxes, predicted_image,  i, delta_x_threshold, slope_threshold_1, slope_threshold_2, col = 'y' )
            # save image with floor line
            floor_img_path =f'{segment_floor_dir}/{filename_0}.jpg'
            cv2.imwrite(floor_img_path, floor_img)
            # save floor info
            list_floor_info.append(list_floor_centroid)
            list_num_floor.append(num_floor)
        # add column 'floor_number','floor_centroids'
        if not (list_num_floor and list_floor_info):
            list_num_floor = np.nan
            list_floor_info = np.nan
        df_building['floor_number'] = list_num_floor
        df_building['floor_centroids'] = list_floor_info

        #  ============ save df_builidng dataframe  ============
        df_building.to_csv(f'{segment_opening_dir}/{filename_0}.csv')

class Model1():
    def __init__(self, WEIGHT_PATH, id_segments, RESULT_PATH, DATASET_ADDRESS, SEGMENT_IMAGE_PATH, **kwargs):
        self.result_path = RESULT_PATH
        self.segment_path = SEGMENT_IMAGE_PATH

        # make_dir(f'./result')
        make_dir(f'{RESULT_PATH}')
        # make_dir(f'{RESULT_FOLDER}/weight')

        # Load model weight
        self.weight_address = WEIGHT_PATH
        self.url = 'https://drive.google.com/uc?id=1-AnJ8XxD4hVQ06ivcabFl8qv94LCSXIc'
        #if not os.path.exists(weight_address):
        #    gdown.download(url, weight_address, quiet=False)

        # add the dict to Cataclog
        # set two categories, opening and building, give their id '0' and '1'.
        self.dict_category_id = {
            'opening': 0,
            'masonry': 1,
            'm6': 1,
            'rcw': 1
        }

        for d in ["train", "val"]:
            DatasetCatalog.register(
                "building_" + d, lambda d=d: get_building_dicts(DATASET_ADDRESS+'/' + d, self.dict_category_id))
            MetadataCatalog.get(
                "building_" + d).set(thing_classes=["opening", "building"])
        self.building_metadata = MetadataCatalog.get("building_train")

        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:

        # === changed parameters ===
        EPOCH = 40
        TOTAL_NUM_IMAGES = 87
        # ===========================

        self.detectron2_cfg = get_cfg()
        self.detectron2_cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.detectron2_cfg.DATASETS.TRAIN = ("building_train",)
        self.detectron2_cfg.DATASETS.TEST = ()
        self.detectron2_cfg.DATALOADER.NUM_WORKERS = 2 # changed parameters
        self.detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        self.detectron2_cfg.SOLVER.IMS_PER_BATCH = 2
        self.detectron2_cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        self.detectron2_cfg.SOLVER.MAX_ITER = int(
            EPOCH*TOTAL_NUM_IMAGES/self.detectron2_cfg.SOLVER.IMS_PER_BATCH-1)
        self.detectron2_cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.detectron2_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # changed parameters
        self.detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # changed parameters

        # weights
        self.detectron2_cfg.MODEL.WEIGHTS = self.weight_address  # path to the model we just trained
        self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80   # set a custom testing threshold

        self.predictor = DefaultPredictor(self.detectron2_cfg)

        # ============ for running model1 to predict building instance and opening   ============
        self.cropped_img_dir = f'{RESULT_PATH}/crop_imgs'
        self.opening_info_dir = f'{RESULT_PATH}/open_info'
        self.floor_dir = f'{RESULT_PATH}/floor_imgs'
        self.test_segments = [item for item in id_segments.split(',')]

    def run(self):

        # ============ run model1 to predict building instance and opening   ============

        for id_segment in self.test_segments:
            dataset_dir = f'{self.segment_path}/{str(id_segment)}'
            print('dataset_dir')
            model1(dataset_dir, self.cropped_img_dir, self.opening_info_dir, self.floor_dir)
            merge_opening_info_csv(
                path=f'{self.opening_info_dir}/{id_segment}', id_segment=id_segment)

        print(f'============ Finish, result is saved at {self.result_path}  ============ ')
        return