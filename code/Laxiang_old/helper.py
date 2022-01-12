import torch, torchvision
assert torch.__version__.startswith("1.8")   # need to manually install torch 1.8 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random, shutil, gdown, warnings

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

from glob import glob
# from google.colab.patches import cv2_imshow
warnings.filterwarnings("ignore")


def make_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)


# ============ create dataset loader of Mask RCNN ============
def get_building_dicts(img_dir, 
    dict_category_id = {
      'opening':0,
      'masonry':1,
      'm6':1,
      'rcw':1
    }):
    # if your dataset is in COCO format, this cell can be replaced by the following three lines:
    # from detectron2.data.datasets import register_coco_instances
    # register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
    # register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    dict_category_id = {
      'opening':0,
      'masonry':1,
      'm6':1,
      'rcw':1
    }
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        # since we use VIA 2.0 tool， annos is a list not a set
        for idx, anno in enumerate(annos):

            # assert not anno["region_attributes"]
            #  record category_id
            category_id = dict_category_id[anno["region_attributes"]['class_name']]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# ============ model 1 opening  ============
def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def get_image_info(origin_image, mask_image):

  x = origin_image.shape[0]
  y = origin_image.shape[1]
  origin_area = x*y
  image = np.zeros((x,y), dtype=np.uint8)  # create blank white image
  image_new = (image+mask_image*255).astype('uint8')  # 
  # find contour
  contours,hierarchy = cv2.findContours(image_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
  # sort contours by area
  if len(contours) > 1:
    contours.sort(key=cnt_area, reverse=True)
  # find centroid
  cnt = contours[0]
  M = cv2.moments(cnt)
  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])
  centroid = (cx,cy)
  # find area
  mask_area = cv2.contourArea(cnt)
  ratio = np.around(mask_area/origin_area,3)

  return centroid, mask_area, origin_area, ratio, image_new, cnt


def get_centroid(origin_image, masks):
  dict_centroid = dict()
  for idx, mask in enumerate(masks):
    res = get_image_info(origin_image, mask)
    centroid = res[0]
    dict_centroid[idx] = centroid
  return dict_centroid


def point_inside_polygon_test(point, cnt_building, image_building):
  #  check if point is inside contour/shape
  result = cv2.pointPolygonTest(cnt_building, point, False)
  return result


def find_opening_insidie_building(origin_image, masks, indexs_building, indexs_opening, df_building):
  list_opening_indexs = []
  list_centroid_building = []
  for idx_building in indexs_building:
    idx_opening_inside_building = []

    mask_building = masks[idx_building]
    res_building = get_image_info(origin_image, mask_image=mask_building)
    cnt_building = res_building[-1] 
    image_building = res_building[-2] 
    
    centroid_building = res_building[0]
    list_centroid_building.append(list(centroid_building))

    for idx_opening in indexs_opening:
      # mask
      mask_opening = masks[idx_opening]
      # find centroid, cnt of building and image
      res_opening = get_image_info(origin_image, mask_image=mask_opening)
      centroid_open  = res_opening[0] 
      # check if opening inside the building 
      inside_res = point_inside_polygon_test(centroid_open, cnt_building, image_building)
      if inside_res==1.0:
        idx_opening_inside_building.append(idx_opening)
    list_opening_indexs.append(idx_opening_inside_building)

  df_building['building_centroid_coordinate'] = list_centroid_building
  df_building['opening_index_inside'] = list_opening_indexs


def get_opening_facade_ratio(origin_image, masks, indexs_building, indexs_opening, df_building):
  list_ratio = []
  for idx, idx_building in enumerate(df_building.building_index.values):
    # building area
    mask_building = masks[idx_building]
    res_building = get_image_info(origin_image, mask_image=mask_building)
    area_building  = res_building[1] 

    # opening indexs inside building 
    list_opening_index = df_building.iloc[idx,:]['opening_index_inside']
    total_opeing_area = 0

    for idx_opening in list_opening_index:
      # find opeing area
      mask_opening = masks[idx_opening]
      res_opening = get_image_info(origin_image, mask_image=mask_opening)
      area_open  = res_opening[1] 
      total_opeing_area+= area_open

    # opening/builiding facade ratio
    ratio = np.around(total_opeing_area/area_building,3)
    list_ratio.append(ratio)
  df_building['opening_facade_ratio'] = list_ratio


def process_output(origin_image, output_image, dict_centroid):
  # resize the prediction output and make it as same size as input image
  predicted_image = output_image.astype(np.uint8)
  predicted_image = cv2.resize(predicted_image, (origin_image.shape[1], origin_image.shape[0]))

  # draw label 
  for label, coordinate in dict_centroid.items():
    cv2.putText(predicted_image,str(label),coordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

  return predicted_image



# ============ merge csv file  ============
def merge_opening_info_csv(path, id_segment):
  save_name = f'{path}/{id_segment}_merged.csv'
  if os.path.exists(save_name):
    print('Already exist merged file')
  else:
    files = sorted(glob(path+'/*.csv')) 
    len_files = len(files)
    if len_files==0:
      print('no csv file')
    elif len_files==1:
      df_merged = pd.read_csv(files[0],encoding='gbk',index_col=0)  
      df_merged.to_csv(save_name)
    else:
      df1 = pd.read_csv(files[0],encoding='gbk',index_col=0)  
      for file in files[1:]:     
          df2 = pd.read_csv(file,encoding='gbk', index_col=0)  
          df1 = pd.concat([df1,df2],axis=0,ignore_index=True)  
      df_merged = df1
      df_merged.to_csv(save_name)

# ============ model2 building centroid  ============
def find_building_centroid(origin_image, masks):
  list_centroid_building = []
  for i,val in enumerate(masks):
    mask_building = masks[i]
    res_building = get_image_info(origin_image, mask_image=mask_building)  
    centroid_building = res_building[0]
    list_centroid_building.append(list(centroid_building))
  return list_centroid_building
