# import some common libraries
import os, cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

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
    
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg"):
            im = cv2.imread(DATASET_DIR+'/'+filename)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=building_metadata, 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            print(outputs)

            img_name = 'inference_on_'+filename
            savepath = TARGET_PATH + img_name

            cv2.imwrite(savepath, out.get_image()[:, :, ::-1])

    return detec_cfg

def draw_instance_predictions(self, predictions):
    """
    Draw instance-level prediction results on an image.

    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
    else:
        masks = None

    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
        colors = [
            self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
        ]
        alpha = 0.8
    else:
        colors = None
        alpha = 0.5

    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.reset_image(
            self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
        )
        alpha = 0.3

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output