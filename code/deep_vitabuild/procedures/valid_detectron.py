# import some common libraries
import os, cv2
import torch

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# import some local lib
from deep_vitabuild.procedures.train_detectron import get_building_dicts

class ValidationLoss(HookBase):
    def __init__(self, detec_cfg):
        super().__init__()
        self.detec_cfg = detec_cfg.clone()
        self.detec_cfg.DATASETS.TRAIN = detec_cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.detec_cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

def add_val_loss(detec_cfg, gen_cfg):
    detec_cfg.DATASETS.VAL = tuple(gen_cfg.VALIDATION.CATALOG)
    os.makedirs(detec_cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(detec_cfg) 
    val_loss = ValidationLoss(detec_cfg)  
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()
    return detec_cfg, trainer

def inference_val(detec_cfg, gen_cfg, building_metadata):
    # Inference should use the config with parameters that are used in training
    # detec_cfg now already contains everything we've set previously. We changed it a little bit for inference:
    detec_cfg.MODEL.WEIGHTS = gen_cfg.VALIDATION.WEIGHTS  # path to the model we just trained
    detec_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = gen_cfg.VALIDATION.SCORE_THRESH_TEST   # set a custom testing threshold
    predictor = DefaultPredictor(detec_cfg)

    #if not os.path.exists('/content/val_predict'):
    val_ouput_path = gen_cfg.VALIDATION.TARGET_PATH
    os.makedirs(val_ouput_path, exist_ok=True)

    
    dataset_dicts = get_building_dicts(gen_cfg.VALIDATION.DATASET_DIR)
    # From here to change the numer of imgs to show
    # num_to_show = 1
    # for d in random.sample(dataset_dicts,num_to_show):  
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=building_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        img_name = '/predict_'+d["file_name"].split('/')[-1]
        savepath = val_ouput_path + img_name
        cv2.imwrite(savepath, out.get_image()[:, :, ::-1])
        #cv2_imshow(out.get_image()[:, :, ::-1])
    
    return detec_cfg

def evaluate_AP(detec_cfg, gen_cfg, trainer):
    print(gen_cfg.VALIDATION.CATALOG[0])
    evaluator = COCOEvaluator(gen_cfg.VALIDATION.CATALOG[0], ("bbox", "segm"), False, output_dir=gen_cfg.VALIDATION.TARGET_PATH)
    val_loader = build_detection_test_loader(detec_cfg, gen_cfg.VALIDATION.CATALOG[0])
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`