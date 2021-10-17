# Code

## Introduciton

This project use deep learning model mask rcnn to detect the builidng instance, openings and building type. Also, builidng duplicaitons are found by using  image clustring method 'DBSDCAN'.

This project is mainly divided into **three models**.

- Model1: First use mask-rcnn model to predict the builiding instance and crop the building instance, then estimate the number of building floor
- Model2: use mask-rcnn model to predict the building type
- Duplicaiton: use DBSCAN clustring to group the builidng dupilcaitons which are predicted by model1, then count the building number, finally combine the building type from model 2 to th building instance.

## Before you run

-  Please download the zip files in the release, unzipped `data.zip` and put this folder to `code/data`
- Move to this project folder

```
cd  /path/to/this_project_folder/
```

- this project need use GPU, please open you CUDA

## Installation

Run the `run.sh` to install packages used in this project

```
./run.sh
```

In this project,  mask rcnn model uses CUDA 10.2 + torch 1.8. If you want to use other cuda or torch version, please see this [ducumentation]() for more instructions

## Data preparation

```
python create_segment_data.py --segments_info_path ./data/original_data/segments_info.csv \
                              --streams_dir ./data/original_data \
                              --segment_image_path ./data/segments \
                              --id_segments '16888'
```

The original building image dataset is saved in the streams. We only concern about the streams taken from the panorama camera direction 1 and direction 4 (left and right side). This code select the relevant images and saved by the segment id

In this repo, we provide two 2 segmemnts for instruction.  Segment 16878 and 16888 are extracted and saced in ` ./data/segments`

**Note:**  If you want to use `create_segment_data.py`, please add relevant streams_dir from Basel dataset into `.data/original_data`. For example, in order to create segment 16888, please add stream 82148 into `.data/original_data`. This realtionship can be found from `dataset/data_basel_images_pc/segments_info.csv`


## Training 

- Model1 Training : building instance and opening

  More detail please read the notebook `training/model1_train.ipynb`

- Model2 Training : building type

  More detail please read the notebook`training/model2_train.ipynb`
## Prediction

For the documentation, please run 

```
python ***.py --help
```

### Model1: building instance and opening

```
python model1.py --weight_path ./data/model_weight/model1_weight.pth \
                 --data_path ./data/training_data/model1 \
                 --segment_image_path ./data/segments \
                 --id_segments '16888' \
                 --result_path ./result/model1
```

- Arguments

  weight_path:  path of model trained weight 

  data_path:  path for train and validation set of model 

  segment_image_path:  path for builidng images which are grouped in semgnets

  id_segments:  testing segment ids. It is a string, ex, '16878,16888,...'

  result_path: paht for saving the model result

- Result

  This model's results is cropped building instances, buildng opening's informantion (openging/facade ratio and building instance centroid coordinate), and builindg floor images

  cropped building instances (width > 500 px) are saved in `./result/model1/crop_imgs`
  
  buildng opening's informantion are saved in `./result/model1/open_info`
  
  builindg floor images are saved in `./result/model1/floor_imgs`
  
  `./result/model1/crop_imgs`contrain the cropped buildng instances. For example, for the image `16888_82148_18_1.jpg`, the predicted builidng instances and builidng floors are at below:

  <div align="center">
    <img width="30%" alt=" building image" src="./readme_imgs/model1.jpg">
    <img width="30%" alt=" building floor" src="./readme_imgs/model1_floor.jpg">
    <img width="30%" alt="cropped building instance" src="./readme_imgs/model1_result.jpg">
  </div>

`		./result/model1/open_info`  contains the building' openign_info csv files. Its form is like this:


|  file_name   | building_index  | building_centroid_coordinate| opening_facade_ratio|
|  ----  | ----  |  ----  | ----  |
| 16888_82148_18_1.jpg | 11 | [1094,90] | 0.214 |


### Model2: building type

```
python model2.py --weight_path ./data/model_weight/model2_weight.pth \
                 --data_path ./data/training_data/model2 \
                 --segment_image_path ./data/segments \
                 --id_segments '16888' \
                 --result_path ./result/model2
```

- Arguments

  same as model 1

- Result

  This model's result is building type.
  
  Predicted building result are saved in `./result/model2/predicted_imgs`
  buildng opening's informantion are saved in `./result/model2/building_info`
  
  The prediction on `16888_82148_18_1.jpg` by using model2
  
  <div align="center">
    <img width="35%" alt=" building image" src="./readme_imgs/model2.jpg">
    <img width="35%" alt="predicted building type" src="./readme_imgs/model2_result.jpg">
  </div>
  
  `./result/model2/building_info`  has csv files containing building type. Its form is like this:

|  file_name   | building_type  |building_centroid_coordinate|
|  ----  | ----  |  ----  |
| 16888_82148_18_1.jpg  | m6 | [1099,901]  | 0.184 |


### Duplication: builidng number

Before running the `duplicaiotn.py`, please insure you are already run the `model1.py` and `model2.py` since duplication model should use the cropped instance from model1 and buidling type form model2.

```
python duplication.py --weight_path ./data/affnet_weight \
                      --segment_cropped_image_path ./result/model1/crop_imgs \
                      --id_segments '16888' \
                      --result_path ./result/duplication \
                      --model1_open_info_folder ./result/model1/open_info \
                      --model2_building_info_folder ./result/model2/building_info
```

- Arguments

  weight_path:  path of affnet pretrained weight 

  segment_cropped_image_path:  path of cropped builidng instance predicted from model1

  model1_open_info_folder: path of model1 opening info result

  model2_building_info_folder: path of model2 building info result

- Result

  This model's result is clustered duplications and building type

  clustered duplications are saved in `./result/duplication/clustered_imgs`
  building types for different cluster are saved in `./result/duplication/cluster_building_type`

  

  For example, one clustered builidng duplciations in the segment 16888.

  <div align="center">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/1.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/2.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/3.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/4.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/5.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/6.jpg">
  </div>
  
  <div align="center">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/7.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/8.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/9.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/10.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/11.jpg">
    <img width="15%" alt="duplication" src="./readme_imgs/duplication/12.jpg">
  </div>
  
  
  
  `./result/duplication/cluster_building_type`  contains the segments' builidng type csv files. Its form is like this: (cluster -1 is a noisy cluster)

|  cluster_name   | building_type  | id_segment |
|  ----  | ----  |  ----  |
| 0  | masonry | 16888 |
| 1  | rcw | 16888 |
| 2  | m6 | 16888 |
| 3  | m6 | 16888 |


**Note**: 
- All the result of 15 testing segments is in the folder `code/segments_result`, since the result's size is too big, it is not uploaded to github. Please ask Alireza to get the result.
- if this repo's predicted segments' building numbers and building type are different from the thesis report, use this code's result since this code's result is the final version and the thesis result is an early version.

## Tutorial Colab Notebook

`run.ipynb`notebook is the tutroial of running this three model. Just change the runtime to GPU model and run it all. Then all the result will be shown and saved.

Step1: use google colab to git clone this repo and change the the runtime to GPU model 

Step2: run the `run.ipynb` notebook to get all results


