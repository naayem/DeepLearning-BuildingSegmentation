import torch, torchvision
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# address of building dataset
DATASET_ADDRESS = '/home/facades/projects/buildings_segmentation_detection/code/data'
OUTPUT_DIR = '/home/facades/projects/buildings_segmentation_detection/output3'

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_building_dicts(img_dir):
    # if your dataset is in COCO format, this cell can be replaced by the following three lines:
    # from detectron2.data.datasets import register_coco_instances
    # register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
    # register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []

    # here we set 'masonry', 'm6', 'rcw' as a same class 'building'
    dict_category_id = {
      'opening':0,
      'masonry':1,
      'm6':1,
      'rcw':1
    }
    print(imgs_anns.values())
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

def add_to_catalog():
    # add the dict to Cataclog
    DatasetCatalog.clear()
    for d in ["totest","train", "val"]:
        DatasetCatalog.register("building_" + d, lambda d=d: get_building_dicts(DATASET_ADDRESS+'/' + d))
        MetadataCatalog.get("building_" + d).set(thing_classes=["opening","building"])
    building_metadata = MetadataCatalog.get("building_train")
    building_val_metadata = MetadataCatalog.get("building_val")
    building_totest_metadata = MetadataCatalog.get("building_totest")
    return building_metadata, building_val_metadata

def visualizer(building_metadata, dataset_dicts):
    dataset_dicts = get_building_dicts(DATASET_ADDRESS+'/train')
    for d in random.sample(dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        cv2_imshow(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2_imshow(out.get_image()[:, :, ::-1])

def cfg_detectron():
# here is the parametes we can change for training
    EPOCH = 40
    TOTAL_NUM_IMAGES = 113

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("building_train", "building_totest",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = int(EPOCH*TOTAL_NUM_IMAGES/cfg.SOLVER.IMS_PER_BATCH-1)     # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 2 class ("opening","masonry","m6","rcw"). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg



class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))
        
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

def train_detectron(cfg):
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    return

def add_val_loss(cfg):
    cfg.DATASETS.VAL = ("building_val",)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    val_loss = ValidationLoss(cfg)  
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg, trainer


def inference_val(cfg, building_metadata):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    #if not os.path.exists('/content/val_predict'):
    os.makedirs('/home/facades/projects/buildings_segmentation_detection/code/data/val_predict', exist_ok=True)

    
    dataset_dicts = get_building_dicts(DATASET_ADDRESS+"/val")
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

        img_name = 'predict_'+d["file_name"].split('/')[-1]
        savepath = '/home/facades/projects/buildings_segmentation_detection/code/data/val_predict/' + img_name
        cv2.imwrite(savepath, out.get_image()[:, :, ::-1])
        #cv2_imshow(out.get_image()[:, :, ::-1])
    
    return cfg

def evaluate_AP(cfg, trainer):
    evaluator = COCOEvaluator("building_val", ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "building_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`

def inference_detectron(cfg, building_metadata):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.40   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    #if not os.path.exists('/content/val_predict'):
    os.makedirs('/data/facades/outputs/inferences/inference_full2/', exist_ok=True)
    DATASET_DIR = '/data/facades/dataset/dataset_complete/dataset/data_basel_images_pc/basel_dataset/Daten_segments1'

    for folder in next(os.walk(DATASET_DIR))[1]:
        folder_path = DATASET_DIR+'/'+folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                print(filename)
                print(folder_path+filename)
                im = cv2.imread(folder_path+'/'+filename)
                outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                v = Visualizer(im[:, :, ::-1],
                            metadata=building_metadata, 
                            scale=0.5, 
                            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                img_name = 'inference_on_'+filename
                savepath = '/data/facades/outputs/inferences/inference_full2/' + img_name
                cv2.imwrite(savepath, out.get_image()[:, :, ::-1])
                #cv2_imshow(out.get_image()[:, :, ::-1])
    
    return cfg

def count_dataset():
    DATASET_DIR = '/data/facades/dataset/dataset_complete/dataset/data_basel_images_pc/basel_dataset/Daten_segments1'
    counter = 0
    for folder in next(os.walk(DATASET_DIR))[1]:
        folder_path = DATASET_DIR+'/'+folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                counter +=1
    
    print('############################COUNTER###################')
    print(counter)
    return

def main():
    building_metadata, building_val_metadata = add_to_catalog()
    dataset_dicts = get_building_dicts(DATASET_ADDRESS+'/train')
    cfg = cfg_detectron()
    print(cfg.OUTPUT_DIR)
    #train_detectron(cfg)
    #cfg, trainer = add_val_loss(cfg)
    #cfg = inference_val(cfg, building_metadata)
    #evaluate_AP(cfg, trainer)

    #inference_detectron(cfg, building_metadata)
    count_dataset()

if __name__ == '__main__':
    main()