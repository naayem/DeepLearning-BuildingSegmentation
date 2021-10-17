# Building instance segementation and Building type detection

Authors: Yaxiong Luo, EE, VITA Lab, EPFL

Supervisors: Saeed Saadatnejad, Alireza Khodaverdian, Prof. Alexandre Massoud Alahi



## Content

`code`: all the code used in this project

`dataset`: introduction of dataset 

`files`: presentation and reports



**Note**: please download the zip files in the release section, unzip `data.zip` and put this folder in `code/data`; unzip `annotated_data.zip` and put this folder in `dataset/annotated_data`;  


## Introduciton

This project use deep learning model mask rcnn to detect the builidng instance, openings and building type. Also, builidng duplicaitons are found by using  image clustring method 'DBSDCAN'.

This project is mainly divided into **three models**.

- Model1: First use mask-rcnn model to predict the builiding instance and crop the building instance, then estimate the number of building floor
- Model2: use mask-rcnn model to predict the building type
- Duplicaiton: use DBSCAN clustring to group the builidng dupilcaitons which are predicted by model1, then count the building number, finally combine the building type from model 2 to th building instance.
# buildings_segmentation_detection
