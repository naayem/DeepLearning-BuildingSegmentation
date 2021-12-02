# import some common libraries
import numpy as np
import os, json, cv2, random, glob
import tqdm, gc
import matplotlib.pyplot as plt

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

YOUR_TOKEN = '3aacbf1f972d9cc0c1a8f2517f0d8e306c91ac8ffe7acad3'  # your token of the web platform
YOUR_DATASET_ID = '6192c2168dae864d8d830838'  # the id of your dataset on the web platform
DATASET_ROOT = '/data/facades/basel_lightly/test'

# allow setting of token and dataset_id from environment variables
def try_get_token_and_id_from_env():
    token = os.getenv('TOKEN', YOUR_TOKEN)
    dataset_id = os.getenv('AL_TUTORIAL_DATASET_ID', YOUR_DATASET_ID)
    return token, dataset_id

def predict_and_overlay(model, filename):
    # helper method to run the model on an image and overlay the predictions
    im = cv2.imread(filename)
    out = model(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(out["instances"].to("cpu"))
    plt.figure(figsize=(16,12))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.tight_layout()

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

def inference_AL(detec_cfg, gen_cfg):
    YOUR_TOKEN, YOUR_DATASET_ID = try_get_token_and_id_from_env()
    print(YOUR_TOKEN)
    print(YOUR_DATASET_ID)

    api_client = ApiWorkflowClient(dataset_id=YOUR_DATASET_ID, token=YOUR_TOKEN)
    al_agent = ActiveLearningAgent(api_client)

    # let's print the first 3 entries
    print(al_agent.query_set[:3])

    print(len(al_agent.query_set))

    predictor = DefaultPredictor(detec_cfg)

    obj_detection_outputs = []
    pbar = tqdm.tqdm(al_agent.query_set)
    for fname in pbar:
        fname_full = os.path.join(gen_cfg.ACTIVE_LEARNING.DATASET_ROOT, fname)
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
        n_samples=100,
        method=SamplingMethod.CORAL,
        name='active-learning-loop-1'
    )
    al_agent.query(config, scorer)