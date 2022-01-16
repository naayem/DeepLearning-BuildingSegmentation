"""
This file takes the coordinates of the openings in an image and calculate 
the coordinates of these openings in the cartesian coordinates and give 
the floor estimation in a cartesian plan
"""
import json
import numpy as np
import cv2 
from math import *
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from .swiss_to_gps_coordinates import swiss_to_gps

def calculate_rotation_matrix(rx,ry,rz):
    first_line = np.array([cos(ry)*cos(rz),-cos(ry)*sin(rz),sin(ry)])
    second_line = np.array([cos(rx)*sin(rz)+sin(rx)*sin(ry)*cos(rz),cos(rx)*cos(rz)-sin(rx)*sin(ry)*sin(rz),-sin(rx)*cos(ry)])
    third_line = np.array([sin(rx)*sin(rz)-cos(rx)*sin(ry)*cos(rz),sin(rx)*cos(rz)+cos(rx)*sin(ry)*sin(rz),cos(rx)*cos(ry)])
    R= np.array([first_line,second_line,third_line])
    return R

# get the information of the image from the json file
def get_info(JS_file,img_name):
    file= open(JS_file)
    data=json.load(file) 
    rx,ry,rz = 0,0,0
    x,y = 0,0
    c,pi = 0,0
    sensorID = ''
    for images in data['configurations']:
        for items in images['images']:
            for item in items['items']:
                if item['imagePath'].split('/')[-1]==img_name.split('/')[-1]:
                    rx = item["rx"]
                    ry = item["ry"]
                    rz = item["rz"]
                    sensorID = item["sensorId"]
                    x = item["x"]
                    y = item["y"]
        for sensors in images['sensorarrays']:
            for sensor in sensors['sensors']:
                if sensor['sensorId'] == sensorID:
                    c = sensor['c']
                    pi = sensor['pixelsize']
    file.close()
    return rx, ry, rz, c, pi, x, y

# Calculate the distance from the camera to specific point 
# You should contact the company that gave us the image to see how to fix this function cause I wrote this function when I they where in holidays and no one answered my email
def calculate_depth(img_name,xi,yi):
    name = img_name.split('/')[-1]
    name = name.split('.')[0]
    folder_name = name.split('-')[0]
    im = cv2.imread('/content/gdrive/MyDrive/new_data/png/'+str(folder_name)+'/'+str(name)+'.png',0) #make sure this is the write location for the depth map
    #return im[int(xi),int(yi)]
    return 12000

# Calculate the location of the sensor coordinates
def from_image_coordinates_to_sensor_coordinates(xi,yi,pi,img_name):
    im = cv2.imread(img_name)
    height, width, chanel = im.shape
    hi, wi = height/2, width/2
    xs = (xi-wi)*pi
    ys = (hi-yi)*pi
    return xs,ys

# Convert the coordinates from perspective into Cartesian Coordinates
def calculate_P(JS_file,img_name,xi,yi):
    rx,ry,rz,c, pi,x,y=get_info(JS_file,img_name)
    R = calculate_rotation_matrix(rx,ry,rz)
    pright,pup = from_image_coordinates_to_sensor_coordinates(xi,yi,pi,img_name)
    p = np.array([pright,pup,-c])
    d = calculate_depth(img_name,xi,yi)
    m = d/c
    res = np.matmul(R,p)
    return m*res

# Uses the file swiss_to_gps_coordinates to find the GPS coordinates of the building
# Once the calculate_depth function will be fixed you should be able to find more accurate results using the distance
def get_gps_coordinates(JS_file,img_name):
    rx,ry,rz,c, pi,x,y=get_info(JS_file,img_name)
    return swiss_to_gps(x+rx,y+ry)

# Draw the projected floor in a Cartesian Plan
def build_plot(group_list, img_name, cur_df_centroid, JS_file,col='y'):
    x_array, y_array = [], []
    x_proj,y_proj=[],[]
    for group_labels in group_list:
        # find group centrods
        group_centroid = cur_df_centroid[cur_df_centroid.label.isin(group_labels)].sort_values(by='y' if col == 'x' else 'x')
        group_centroid['centroid'] = group_centroid.apply(lambda df: (df['x'], df['y']), axis=1)
        # take the coordinates of the centroids
        c = group_centroid.centroid.values
        first,snd = zip(*c)
        x_array = np.append(x_array,first)
        y_array = np.append(y_array,snd)
        # Convert into cartesian coordinates
        for i in range(0,x_array.size):
            a= calculate_P(JS_file, img_name,x_array[i]/2,y_array[i]/2)
            x_proj= np.append(x_proj,a[0])
            y_proj= np.append(y_proj,a[1])
        # Plot the points and draw the floor line 
        # Here the floor line is a best fit line and not a connection of the points
        for i in range(0,x_array.size):
            plt.plot([x_proj[i]],[y_proj[i]], marker='o', markersize=3, color='red', label='point')
        slope, intercept = np.polyfit(x_proj, y_proj, 1)
        plt.plot(x_proj,intercept + slope * x_proj)
        
    plt.savefig('/content/result/'+img_name.split('/')[-1])


#TEST
"""
glist = [[2, 9.0, 4.0], [3, 7.0, 0.0]]
name = "/content/gdrive/MyDrive/new_data/jpg/14633/14633-30-0.jpg"
df = pd.DataFrame(np.array([[0, 0, 115], [2, 2, 517], [3, 3, 117],[4, 4, 520], [7, 7, 116], [9, 9, 520]]))
df.columns = ['label', 'x', 'y']
build_plot(glist,name,df,'/content/semester-project/code/data/JS.json')
"""