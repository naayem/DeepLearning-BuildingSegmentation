# imports for lightly
from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput

from imantics import Mask
from rdp import rdp

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

def convert_annot_detecton2via_RDP(filename, outputs, size, rdp_epsilon):
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
            list_x = list_poly[i][0][0::2]
            list_y = list_poly[i][0][1::2]
            #list_x = list_x[0::max(len(list_x)//15,1)]
            #list_y = list_y[0::max(len(list_y)//15,1)]
        else:
            region["region_attributes"]["class_name"] = 'm6'
            list_x = list_poly[i][0][0::2]
            list_y = list_poly[i][0][1::2]
            #list_x = list_x[0::max(len(list_x)//30,1)]
            #list_y = list_y[0::max(len(list_y)//30,1)]

        rdp_format = []
        rdp_format_coord = []
        for x,y in zip(list_x, list_y):
            rdp_format_coord.append(x)
            rdp_format_coord.append(y)
            rdp_format.append(rdp_format_coord)
            rdp_format_coord=[]
        smpl_poly = rdp(rdp_format, epsilon=rdp_epsilon)
        list_x = [coord[0] for coord in smpl_poly]
        list_y = [coord[1] for coord in smpl_poly]
        
        for x,y in zip(list_x, list_y):
            region["shape_attributes"]["all_points_x"].append(x)
            region["shape_attributes"]["all_points_y"].append(y)
        annotation[f"{filename}{size}"]['regions'].append(region)

    return annotation

def convert_bbox_detectron2lightly(outputs):
    # convert detectron2 predictions into lightly format
    height, width = outputs['instances'].image_size
    boxes = []

    for (bbox_raw, score, class_idx) in zip(outputs['instances'].pred_boxes.tensor,
                                            outputs['instances'].scores,
                                            outputs['instances'].pred_classes):
        x0, y0, x1, y1 = bbox_raw.cpu().numpy()
        x0 /= width
        y0 /= height
        x1 /= width
        y1 /= height

        boxes.append(BoundingBox(x0, y0, x1, y1))
    output = ObjectDetectionOutput.from_scores(
      boxes, outputs['instances'].scores.cpu().numpy(),
      outputs['instances'].pred_classes.cpu().numpy().tolist())
    return output

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
            list_x = list_poly[i][0][0::2]
            list_y = list_poly[i][0][1::2]
            list_x = list_x[0::max(len(list_x)//15,1)]
            list_y = list_y[0::max(len(list_y)//15,1)]
        else:
            region["region_attributes"]["class_name"] = 'm6'
            list_x = list_poly[i][0][0::2]
            list_y = list_poly[i][0][1::2]
            list_x = list_x[0::max(len(list_x)//30,1)]
            list_y = list_y[0::max(len(list_y)//30,1)]

        for x,y in zip(list_x, list_y):
            region["shape_attributes"]["all_points_x"].append(x)
            region["shape_attributes"]["all_points_y"].append(y)
        annotation[f"{filename}{size}"]['regions'].append(region)

    return annotation