# import some common libraries
import os, cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

import pycocotools.mask as mask_util
import numpy as np
from imantics import Polygons, Mask
import imantics
import json

def inference_detectron_full(detec_cfg, gen_cfg, building_metadata):
    DATASET_DIR = gen_cfg.INFERENCE.DATASET_PATH
    TARGET_PATH = gen_cfg.INFERENCE.TARGET_PATH

    # Inference should use the config with parameters that are used in training
    # detec_cfg now already contains everything we've set previously. We changed it a little bit for inference:
    detec_cfg.MODEL.WEIGHTS = gen_cfg.INFERENCE.WEIGHTS  # path to the model we just trained
    detec_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = gen_cfg.INFERENCE.SCORE_THRESH_TEST   # set a custom testing threshold
    predictor = DefaultPredictor(detec_cfg)

    os.makedirs(TARGET_PATH, exist_ok=True)
    
    print(next(os.walk(DATASET_DIR))[1])

    for folder in next(os.walk(DATASET_DIR))[1]:
        folder_path = DATASET_DIR+'/'+folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                im = cv2.imread(folder_path+'/'+filename)
                outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                v = Visualizer(im[:, :, ::-1],
                            metadata=building_metadata, 
                            scale=0.5, 
                            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                img_name = 'inference_on_'+filename
                savepath = TARGET_PATH + img_name
                cv2.imwrite(savepath, out.get_image()[:, :, ::-1])
    
    return detec_cfg

def inference_detectron_folder(detec_cfg, gen_cfg, building_metadata):
    DATASET_DIR = gen_cfg.INFERENCE.DATASET_PATH
    TARGET_PATH = gen_cfg.INFERENCE.TARGET_PATH

    # Inference should use the config with parameters that are used in training
    # detec_cfg now already contains everything we've set previously. We changed it a little bit for inference:
    detec_cfg.MODEL.WEIGHTS = gen_cfg.INFERENCE.WEIGHTS  # path to the model we just trained
    detec_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = gen_cfg.INFERENCE.SCORE_THRESH_TEST   # set a custom testing threshold
    predictor = DefaultPredictor(detec_cfg)

    
    os.makedirs(TARGET_PATH, exist_ok=True)
    
    via_dict = {}
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg"):
            size = os.path.getsize(DATASET_DIR+'/'+filename)
            im = cv2.imread(DATASET_DIR+'/'+filename)

            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=building_metadata, 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
            output_via = convert_annot_detecton2via(filename, outputs, size)
            via_dict.update(output_via)

            img_name = 'inference_on_'+filename
            savepath = TARGET_PATH + img_name

            cv2.imwrite(savepath, out.get_image()[:, :, ::-1])

    jsonpath = TARGET_PATH + 'data.json'
    wrapped = wrap_jsonVia(via_dict)
    with open(jsonpath, 'w') as fp:
        json.dump(wrapped, fp,  indent=4)

    return detec_cfg

def convert_annot_detecton2via(filename, outputs, size):
    annotation = {f"{filename}{size}": {'filename': filename, 'size': size, 'regions': [], 'file_attributes':{} } }

    classes = outputs["instances"].to("cpu").pred_classes.numpy()
    masko = outputs["instances"].to("cpu").pred_masks

    list_poly = []
    for mask in masko:
        mask = mask.numpy()
        polygons = Mask(mask).polygons()
        list_poly.append(polygons.segmentation)
    
    for i in range(len(classes)):
        region = {"shape_attributes": {"name": 'polygon', "all_points_x": [], "all_points_y": []}, "region_attributes":{"class_name": ''}}
        if classes[i] == 0:
            region["region_attributes"]["class_name"] = 'opening'
        else:
            region["region_attributes"]["class_name"] = 'm6'

        j = 1
        for coordinate in list_poly[i][0]:
            if not (j%20-1):
                region["shape_attributes"]["all_points_x"].append(coordinate)
            if not (j%20):
                region["shape_attributes"]["all_points_y"].append(coordinate)
            j = j+1
        annotation[f"{filename}{size}"]['regions'].append(region)

    return annotation

def wrap_jsonVia(images_annotations):
    wrapping = {
                "_via_settings": {                # settings used by the VIA application
                    "ui": {
                    "annotation_editor_height": 25,
                    "annotation_editor_fontsize": 0.8,
                    "leftsidebar_width": 18,
                    "image_grid": {
                        "img_height": 80,
                        "rshape_fill": "none",
                        "rshape_fill_opacity": 0.3,
                        "rshape_stroke": "yellow",
                        "rshape_stroke_width": 2,
                        "show_region_shape": True,
                        "show_image_policy": "all"
                    },
                    "image": {
                        "region_label": "__via_region_id__",
                        "region_color": "__via_default_region_color__",
                        "region_label_font": "10px Sans",
                        "on_image_annotation_editor_placement": "NEAR_REGION"
                    }
                    },
                    "core": {
                    "buffer_size": 18,
                    "filepath": {},
                    "default_filepath": ""
                    },
                    "project": {
                    "name": "via_project_16Feb2021_13h17m"
                    }
                },
                "_via_img_metadata":{ 
                    },
                "_via_attributes":{
                    "region":{
                        "classe names":{
                            "type":"dropdown",
                            "description":"",
                            "options":{
                                "m6":"",
                                "rcw":"",
                                "opening":"",
                                "masonry":""},
                            "default_options":{}
                        }
                    },
                    "file":{}
                },
                "_via_data_format_version": "2.0.10",
                "_via_image_id_list": [             # this contains the list of image-id present in the "_via_img_metadata" dictionary
                ]
            }
    for via_image_id in images_annotations.keys():
        wrapping["_via_image_id_list"].append(via_image_id)

    wrapping["_via_img_metadata"] = images_annotations
    return wrapping