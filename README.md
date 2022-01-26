# Building instance segementation and Building type detection
Authors: Vincent Naayem, MT, VITA Lab, EPFL

Supervisors: Saeed Saadatnejad, Alireza Khodaverdian, Prof. Alexandre Massoud Alahi

## Step by step tutorial

Install everything:
-	Poetry install, to install and resolve all dependencies in pyproject.toml
-	pip install pandas
-	pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

Manage datasets and paths:
- Full dataset: '/data/facades/dataset/dataset_complete/dataset/data_basel_images_pc/basel_dataset/Daten_segments1'
- Training and validation datasets : project_path/code/data
- Output datasets: 'project_path/data’


To run the code: 
- python code/main.py --cfg experiments/experiment.yaml

To check that configurations runs without error:
- In experiment.yaml put all status to 0 and run code

To check that training runs correctly, in the configuration yaml modify accordingly:
- Training: Dataset_dir is the path to the folder containing all the folders of the datasets with their respective annotations.
- Training: TARGET_PATH is the path where any output will be saved: ‘project_path/output/OUTPUT_DIR/EXP_NAME/train’
- Training: CATALOG: name of the dataset to train with ‘building_train’ for ‘train’ folder or ‘building_totest’ for ‘totest’ folder.

To check that Validation runs correctly, in the configuration yaml modify accordingly:
- Validation: Dataset_dir is the path containing the validation dataset folder and its annotations. ‘project_path/code/data/val'
- Validation: TARGET_PATH is the path where any output will be saved: ‘project_path/output/OUTPUT_DIR/EXP_NAME/val’
- Validation: WEIGHTS: Where the wheights of a training we want to test lies. For example: ‘project_path/output/OUTPUT_DIR/EXP_NAME/train/model_final.pth’

Active_learninng:
- DATASET_ROOT : '/data/facades/dataset/dataset_complete/dataset/data_basel_images_pc/basel_dataset/Daten_segments1' #For Full dataset

- DATASET_PATH: 'project_path/data/AL_TEST' # Images to read For annotations
- TARGET_PATH : 'project_path/data/AL_TEST' #For outputs, transfer, annotations
- Active learning annotations resides target_path
- Weights: same as valid
- In Sampler: al_loop: please change from 'active-learning-loop-2' to ‘AL-test’ for example

If everything works fine self-annotated images to annotate resides in TARGET_PATH



## Content

`code`: all the code used in this project

`dataset`: introduction of dataset 

`experiments`: .yaml files with all configurations needed to run the code

`files`: presentation and reports

`output`: output directory for any outputs from the code

`Laxiang_old`: code from previous project


**Note**: please download the zip files in the release section, unzip `data.zip` and put this folder in `code/data`; unzip `annotated_data.zip` and put this folder in `dataset/annotated_data`;  


## Introduction

This project use deep learning model mask rcnn to detect the builidng instance, openings and building type.

This project is mainly divided into **six tasks**:

- Segments transfer
- Training
- Validation
- Inferences
- Active Learning
- Floor Detection


#################################

## The core of the project is a custom yet functional approach to detectron2.

Aims at making detecton2 application more handy and clear.

It comprises of experiment description (.yaml dictionary files), starting point file (`main.py`) and the `deep_vitabuild` directory. The latter includes:

- `model1`: All functions needed for Floor Detections
- `procedures`: All functions needed for Training, Validation, Inference, Active Learning
- `utils`: Additional utilities, e.g. visualization, detectron to via conversions etc..

To use Detectron2 please refer to documentations and tutorials online.
To use Lightly platform please refer to documentations and tutorials online

To run the code:
```
python code/main.py --cfg experiments/experiment.yaml 
```

## Installation

Run the `pyproject.toml` with poetry to install packages used in this project. Refer to poetry Python dependency management and packaging documentations and tutorials

You need to install the missing libraries with pip install.
You can find the pip install command for detectron in Laxiang_old/run.sh

## Data preparation
In yaml file this dictionnary is needed:
```
SEGMENTS_TRANSFER:
  status : 0
  segments_info_path : '/data/facades/dataset/dataset_complete/dataset/data_basel_images_pc/segments_info.csv' #help="csv file to show the relation between segments and streams"
  streams_dir : '/data/facades/dataset/build_seg_annotation1' #help="orginal building images are grouped by streams"
  segment_image_path : '/data/facades/dataset/annotated_data/basel_annotation1' #"Destination: copy to path only select the direction 1 and 4 from the panorama camera and grouped by segments"
  id_segments : '14051, 14052, 14053, 14054, 14055, 14056, 14057, 14058, 14064, 14065, 14066, 14067, 14068, 14069, 14070, 14071, 14072, 16835, 16836, 16837, 16838, 16839, 16849, 16850, 16851, 16852, 16853, 16854, 16855, 16856, 16857, 16858, 16872, 16873, 16875, 16876, 16877, 16878, 16879, 16880, 16881, 16882, 16883, 16884, 16885, 16888, 16889, 16890, 16948, 16949, 19416' #help="segments' id for testing(delimited list input)"
```

The original building image dataset is saved in the streams. We only concern about the streams taken from the panorama camera direction 1 and direction 4 (left and right side). This code select the relevant images and saved by the segment id

In this repo, we provide many segmemnts for instruction.  Segment 16878 and 16888 are extracted and saced in ` ./data/segments`

**Note:**  If you want to use `create_segment_data.py`, please add relevant streams_dir from Basel dataset into `.data/original_data`. For example, in order to create segment 16888, please add stream 82148 into `.data/original_data`. This realtionship can be found from `dataset/data_basel_images_pc/segments_info.csv`

## Training 
In yaml file this dictionnary is needed:
```
TRAINING:
  status : 0
  DATASET_DIR : '/home/facades/projects/buildings_segmentation_detection/code/data' #Where the data for training lies
  TARGET_PATH : '/home/facades/projects/buildings_segmentation_detection/output/week14/building_basel_annotation1/train' # Where to we save outputs
  CATALOG : # Training folders selected, see how catalogs works
    - "building_train"
    - "building_basel_annotation1"
```

## Validation 
In yaml file this dictionnary is needed:
```
VALIDATION:
  status : 0
  DATASET_DIR : '/home/facades/projects/buildings_segmentation_detection/code/data/val' #Where the data for validation lies
  TARGET_PATH : '/home/facades/projects/buildings_segmentation_detection/output/week14/building_basel_annotation1/val' # Where to we save outputs
  CATALOG : # Validation folders selected, see how catalogs works
    - "building_val"
  SCORE_THRESH_TEST : 0.90 
  WEIGHTS : '/home/facades/projects/buildings_segmentation_detection/output/week14/building_basel_annotation1/train/model_final.pth'
  #For Train+val in one go use the output dir where the weights will located after training
```
## Inferences 
In yaml file this dictionnary is needed:
```
INFERENCE:
  status : 0
  STRUCTURE : 'folder' #How to read the dataset. For now: full or folder (through folders and directly thrg items resp)
  DATASET_PATH: '/home/facades/projects/buildings_segmentation_detection/code/data/vTestJson'
  TARGET_PATH : '/home/facades/projects/buildings_segmentation_detection/output/inference_jsonTESTS/'
  SCORE_THRESH_TEST : 0.90
  WEIGHTS : '/home/facades/projects/buildings_segmentation_detection/output/annotation1Plus/model_final.pth' #For Train+inf in one go use the output dir where the weights will located after training
```
## Active Learning 
In yaml file this dictionnary is needed:
```
ACTIVE_LEARNING:
  status : 1
  DATASET_ROOT : '/data/facades/dataset/dataset_complete/dataset/data_basel_images_pc/basel_dataset/Daten_segments1' #For Full dataset
  DATASET_PATH: '/home/facades/projects/buildings_segmentation_detection/data/AL_TEST' # Images to read For annotations
  TARGET_PATH : '/home/facades/projects/buildings_segmentation_detection/data/AL_TEST' #For outputs, transfer, annotations
  SCORE_THRESH_TEST : 0.90
  WEIGHTS : '/home/facades/projects/buildings_segmentation_detection/output/week14/active-learning-loop-1/model_final.pth' #For Train+inf in one go use the output dir where the weights will located after training
  rdp_epsilon: 20
  SAMPLER:
    al_loop: 'active-learning-loop-2' #name of tag in lightly
    n_samples: 100
    method: 'CORAL'
# If you want to train, validate.. and active learning in one go, weights, target_path and dataset_path must match
```

## Active Learning 
In yaml file this dictionnary is needed:
```
FLOOR_DETECTION:
  status : 0
  data_path: '/home/facades/projects/buildings_segmentation_detection/code/data/train'
  segment_image_path: '/home/facades/projects/buildings_segmentation_detection/code/data/totest'
  id_segments: '20200224'
  results_path: '/home/facades/projects/buildings_segmentation_detection/output/RAYMOND/FLOOR'
  WEIGHTS : '/home/facades/projects/buildings_segmentation_detection/output/week14/active-learning-loop-1/model_final.pth' #For Train+inf in one go use the output dir where the weights will located after training
```