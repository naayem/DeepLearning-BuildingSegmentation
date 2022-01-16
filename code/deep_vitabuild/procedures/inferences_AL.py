# import some common libraries
import numpy as np
import os, json, cv2, random, glob
import tqdm, gc
import matplotlib.pyplot as plt
import json

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# imports for lightly
import lightly
from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput
from lightly.active_learning.scorers import ScorerObjectDetection
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.active_learning.agents import ActiveLearningAgent
from lightly.active_learning.config import SamplerConfig
from lightly.openapi_generated.swagger_client import SamplingMethod

from ..utils.detectron2via import wrap_jsonVia, convert_annot_detectron2via_RDP, convert_annot_detectron2via, convert_bbox_detectron2lightly

'''
def try_get_token_and_id_from_env():
    # allow setting of token and dataset_id from environment variables
    token = os.getenv('TOKEN', YOUR_TOKEN)
    dataset_id = os.getenv('AL_TUTORIAL_DATASET_ID', YOUR_DATASET_ID)
    return token, dataset_id
    '''

def inference_AL(detec_cfg, gen_cfg):
    YOUR_TOKEN = gen_cfg.YOUR_TOKEN
    YOUR_DATASET_ID = gen_cfg.YOUR_DATASET_ID

    DATASET_ROOT = gen_cfg.ACTIVE_LEARNING.DATASET_ROOT
    TARGET_PATH = gen_cfg.ACTIVE_LEARNING.TARGET_PATH

    sampler_cfg = gen_cfg.ACTIVE_LEARNING.SAMPLER

    api_client = ApiWorkflowClient(dataset_id=YOUR_DATASET_ID, token=YOUR_TOKEN)
    al_agent = ActiveLearningAgent(api_client)

    # let's print the first 3 entries
    print(al_agent.query_set[:3])
    print(len(al_agent.query_set))

    predictor = DefaultPredictor(detec_cfg)

    obj_detection_outputs = []
    pbar = tqdm.tqdm(al_agent.query_set)
    for fname in pbar:
        fname_full = os.path.join(DATASET_ROOT, fname)
        im = cv2.imread(fname_full)
        out = predictor(im)
        obj_detection_output = convert_bbox_detectron2lightly(out)
        obj_detection_outputs.append(obj_detection_output)

    scorer = ScorerObjectDetection(obj_detection_outputs)
    scores = scorer.calculate_scores()

    max_score = scores['uncertainty_margin'].max()
    idx = scores['uncertainty_margin'].argmax()
    print(f'Highest uncertainty_margin score found for idx {idx}: {max_score}')


    config = SamplerConfig(
        n_samples = sampler_cfg.n_samples,
        method = SamplingMethod.CORAL,
        name = sampler_cfg.al_loop
    )
    al_agent.query(config, scorer)

    list_of_files = get_filenames_in_tag(api_client, sampler_cfg.al_loop)

    transfer_to_target_dir(list_of_files, DATASET_ROOT, TARGET_PATH)
    # Now the files resides in TARGET_PATH, next step in main will get annotations and dump json file in TARGET_PATH

def get_images_directly_from_tagname(gen_cfg):
    YOUR_TOKEN = gen_cfg.YOUR_TOKEN
    YOUR_DATASET_ID = gen_cfg.YOUR_DATASET_ID

    DATASET_ROOT = gen_cfg.ACTIVE_LEARNING.DATASET_ROOT
    TARGET_PATH = gen_cfg.ACTIVE_LEARNING.TARGET_PATH

    sampler_cfg = gen_cfg.ACTIVE_LEARNING.SAMPLER

    api_client = ApiWorkflowClient(dataset_id=YOUR_DATASET_ID, token=YOUR_TOKEN)
    al_agent = ActiveLearningAgent(api_client)

    list_of_files = get_filenames_in_tag(api_client, sampler_cfg.al_loop)

    transfer_to_target_dir(list_of_files, DATASET_ROOT, TARGET_PATH)
    # Now the files resides in TARGET_PATH, next step in main will get annotations and dump json file in TARGET_PATH


def inference_detectron_get_notations(detec_cfg, gen_cfg, building_metadata):
    '''
        Infer predictions on ACTIVE_LEARNING.DATASET_PATH and convert these to annotations to via format
        Note that via format for reading a file with via is not the same that of the json output of via:
            For via 2.0 to read a file a layer is added by wrap_jsonVia()
        Outputs: images_annotations_wrapped.json
        TODO: Suppress polygons with less than 4 points otherwise these polygons will generate problems later
    '''
    DATASET_DIR = gen_cfg.ACTIVE_LEARNING.DATASET_PATH
    TARGET_PATH = gen_cfg.ACTIVE_LEARNING.TARGET_PATH
    rdp_epsilon = gen_cfg.ACTIVE_LEARNING.rdp_epsilon

    # Inference should use the config with parameters that are used in training
    # detec_cfg now already contains everything we've set previously. We changed it a little bit for inference:
    detec_cfg.MODEL.WEIGHTS = gen_cfg.ACTIVE_LEARNING.WEIGHTS  # path to the model we just trained
    detec_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = gen_cfg.ACTIVE_LEARNING.SCORE_THRESH_TEST   # set a custom testing threshold
    predictor = DefaultPredictor(detec_cfg)
    
    os.makedirs(TARGET_PATH, exist_ok=True)
    
    images_annotations = dict()
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg"):
            im = cv2.imread(DATASET_DIR+'/'+filename)
            size = os.path.getsize(DATASET_DIR+'/'+filename)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=building_metadata, 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            image_annotations = convert_annot_detectron2via_RDP(filename, outputs, size, rdp_epsilon)
            images_annotations.update(image_annotations)

    images_annotations_wrapped = wrap_jsonVia(images_annotations)

    with open(TARGET_PATH+'/images_annotations_wrapped.json', 'w') as fp:
        json.dump(images_annotations_wrapped, fp, indent=4)

    return detec_cfg

def inference_detectron_get_notations_report(detec_cfg, gen_cfg, building_metadata):
    '''
        Infer predictions on ACTIVE_LEARNING.DATASET_PATH and convert these to annotations to via format
        Note that via format for reading a file with via is not the same that of the json output of via:
            For via 2.0 to read a file a layer is added by wrap_jsonVia()
        Outputs: images_annotations_imantics_wrapped.json (optional), images_annotations_wrapped.json
        TODO: Suppress polygons with less than 4 points otherwise these polygons will generate problems later
    '''
    DATASET_DIR = gen_cfg.ACTIVE_LEARNING.DATASET_PATH
    TARGET_PATH = gen_cfg.ACTIVE_LEARNING.TARGET_PATH
    rdp_epsilon = gen_cfg.ACTIVE_LEARNING.rdp_epsilon

    # Inference should use the config with parameters that are used in training
    # detec_cfg now already contains everything we've set previously. We changed it a little bit for inference:
    detec_cfg.MODEL.WEIGHTS = gen_cfg.ACTIVE_LEARNING.WEIGHTS  # path to the model we just trained
    detec_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = gen_cfg.ACTIVE_LEARNING.SCORE_THRESH_TEST   # set a custom testing threshold
    predictor = DefaultPredictor(detec_cfg)

    
    os.makedirs(TARGET_PATH, exist_ok=True)
    
    images_annotations = dict()
    images_annotations_imantics = dict()
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg"):
            im = cv2.imread(DATASET_DIR+'/'+filename)
            size = os.path.getsize(DATASET_DIR+'/'+filename)
            print(size)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=building_metadata, 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            image_annotations = convert_annot_detectron2via_RDP(filename, outputs, size, rdp_epsilon)
            image_annotations_imantics = convert_annot_detectron2via(filename, outputs, size)

            images_annotations.update(image_annotations)
            images_annotations_imantics.update(image_annotations_imantics)
            


            img_name = '/inference_on_'+filename
            savepath = TARGET_PATH + img_name

            cv2.imwrite(savepath, out.get_image()[:, :, ::-1])

    images_annotations_wrapped = wrap_jsonVia(images_annotations)
    images_annotations_imantics_wrapped = wrap_jsonVia(images_annotations_imantics)

    with open(TARGET_PATH+'/images_annotations_wrapped.json', 'w') as fp:
        json.dump(images_annotations_wrapped, fp, indent=4)

    with open(TARGET_PATH+'/images_annotations_imantics_wrapped.json', 'w') as fp:
        json.dump(images_annotations_imantics_wrapped, fp, indent=4)

    return detec_cfg

def get_filenames_in_tag(api_client, tag_name):
    '''
        Query the the list of filenames
        Returns a list of filenames 
    '''
    for dict_tagdata in api_client.get_all_tags():
        if dict_tagdata.name == tag_name:
            target_tagdata = dict_tagdata
    return api_client.get_filenames_in_tag(tag_data= target_tagdata)

def transfer_to_target_dir(list_filenames, DATASET_ROOT, TARGET_PATH):
    pbar = tqdm.tqdm(list_filenames)
    for fname in pbar:
        fname_full = os.path.join(DATASET_ROOT, fname)
        im = cv2.imread(fname_full)
        img_name = os.path.basename(fname_full)
        savepath = TARGET_PATH + '/' + img_name
        cv2.imwrite(savepath, im)