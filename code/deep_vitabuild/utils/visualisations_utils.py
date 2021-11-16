# import some common libraries
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer


def visualizer(building_metadata, dataset_dicts):
    DATASET_ADDRESS = ''
    dataset_dicts = get_building_dicts(DATASET_ADDRESS+'/train')
    for d in random.sample(dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        cv2_imshow(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2_imshow(out.get_image()[:, :, ::-1])