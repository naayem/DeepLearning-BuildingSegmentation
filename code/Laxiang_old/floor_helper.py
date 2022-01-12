import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import cv2
import random
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from math import *

# ============ model 1 draw floor line  ============
def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

def get_image_info(origin_image, mask_image):

    x = origin_image.shape[0]
    y = origin_image.shape[1]
    origin_area = x*y
    image = np.zeros((x, y), dtype=np.uint8)  # create blank white image
    image_new = (image+mask_image*255).astype('uint8')  #
    # cv2_imshow(image_new)
    # find contour
    contours, hierarchy = cv2.findContours(
        image_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # sort contours by area
    if len(contours) > 1:
        contours.sort(key=cnt_area, reverse=True)
    # find centroid
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx, cy)
    # find area
    mask_area = cv2.contourArea(cnt)
    ratio = np.around(mask_area/origin_area, 3)

    return centroid, mask_area, origin_area, ratio, image_new, cnt


def get_building_centroid(df_building, building_index, dict_centroid):

    df = df_building.loc[:, ['building_index', 'opening_index_inside']]
    # centroid tabel
    df_centroid = pd.DataFrame(dict_centroid).T
    df_centroid.reset_index(inplace=True)
    df_centroid.columns = ['label', 'x', 'y']

    df_test = df[df.building_index == building_index]  # frame
    cur_list_labels = df_test.opening_index_inside.values[0]  # list

    if len(cur_list_labels) == 0:
        print("No opening is detected")
    cur_df_centroid = df_centroid[df_centroid.label.isin(
        cur_list_labels)]  # DataFrame

    return cur_df_centroid


def crop_image(boxes, building_index, predicted_image):
    cur_box = boxes.astype('int')[building_index]
    x1, y1, x2, y2 = cur_box[0], cur_box[1], cur_box[2], cur_box[3],
    cropped_img = predicted_image[y1:y2, x1:x2]
    bounding_box_min = [x1, y1]
    y_range = y2-y1
    return cropped_img, bounding_box_min, y_range


def find_left_corner(df):
    left_label = df.sort_values('x').iloc[0, :].label
    df_label = df[df.label == left_label]
    return df_label, left_label


def find_slope_right(df_added_label, df_closet, group, x_0, y_0, bounding_box_y_range, delta_x_threshold, slope_threshold_1, slope_threshold_2):

    row_len = df_closet.shape[0]

    if row_len == 0:
        return -1
    df_closet['distance'] = df_closet.apply(
        lambda x: cal_distance(x, x_0, y_0), axis=1)
    df_closet.sort_values('distance', inplace=True)

    added_label = -1
    for i in range(row_len):
        # calculate the slope
        delta_x = df_closet.iloc[i].x - df_added_label.x.values[0]
        delta_y = df_added_label.y.values[0] - df_closet.iloc[i].y

        if abs(delta_y) >= 1/4*bounding_box_y_range:
            added_label = -1
            continue
        slope = delta_y/delta_x
        # print('delta_x',delta_x)
        # print('delta_y',delta_y)
        # print('slope',slope)

        if delta_x > delta_x_threshold:
            if (slope > slope_threshold_2[0]) and (slope < slope_threshold_2[1]):

                added_label = df_closet.iloc[i].label
                group.append(added_label)
                return added_label
        else:
            if (slope > slope_threshold_1[0]) and (slope < slope_threshold_1[1]):

                added_label = df_closet.iloc[i].label
                group.append(added_label)
                return added_label
    return added_label


def find_slope_left(df_added_label, df_closet, group, x_0, y_0, bounding_box_y_range, delta_x_threshold, slope_threshold_1, slope_threshold_2):

    row_len = df_closet.shape[0]
    if row_len == 0:
        return -1

    df_closet['distance'] = df_closet.apply(
        lambda x: cal_distance(x, x_0, y_0), axis=1)
    df_closet.sort_values('distance', inplace=True)

    added_label = -1
    for i in range(row_len-1, -1, -1):
        # calculate the slope
        delta_x = df_added_label.x.values[0] - df_closet.iloc[i].x
        delta_y = df_closet.iloc[i].y - df_added_label.y.values[0]
        slope = delta_y/delta_x

        if abs(delta_y) >= 1/4*bounding_box_y_range:
            added_label = -1
            continue

        if delta_x > delta_x_threshold:
            if (slope > slope_threshold_2[0]) and (slope < slope_threshold_2[1]):
                added_label = df_closet.iloc[i].label
                group.append(added_label)
                return added_label
        else:
            if (slope > slope_threshold_1[0]) and (slope < slope_threshold_1[1]):
                added_label = df_closet.iloc[i].label
                group.append(added_label)
                return added_label
        return added_label


def cal_distance(df, x_0, y_0):
    return sqrt((df.x-x_0)**2+(df.y-y_0)**2)


def find_group_member(df_left_corner, df, group, bounding_box_y_range, delta_x_threshold, slope_threshold_1, slope_threshold_2):

    added_label = df_left_corner.label.values[0]
    while added_label != -1 and added_label != None:
        # print('added_label',added_label)
        df_added_label = df[df.label == added_label]
        x_0, y_0 = df_added_label.x.values[0], df_added_label.y.values[0]
        df_closet_sorted_right = df[df.x > x_0]
        added_label = find_slope_right(df_added_label, df_closet_sorted_right, group, x_0,
                                       y_0, bounding_box_y_range, delta_x_threshold, slope_threshold_1, slope_threshold_2)

    added_label = df_left_corner.label.values[0]
    while added_label != -1 and added_label != None:
        # print('added_label',added_label)
        df_added_label = df[df.label == added_label]
        x_0, y_0 = df_added_label.x.values[0], df_added_label.y.values[0]
        df_closet_sorted_left = df[df.x < x_0]
        added_label = find_slope_left(df_added_label, df_closet_sorted_left, group, x_0, y_0,
                                      bounding_box_y_range, delta_x_threshold, slope_threshold_1, slope_threshold_2)


def create_group(df, bounding_box_y_range, delta_x_threshold, slope_threshold_1, slope_threshold_2):
    df_left_corner, left_corner_label = find_left_corner(df)
    # print('left_corner_label',left_corner_label)
    group = []
    group.append(left_corner_label)
    find_group_member(df_left_corner, df, group, bounding_box_y_range,
                      delta_x_threshold, slope_threshold_1, slope_threshold_2)
    return group


def find_group_list(df, bounding_box_y_range, delta_x_threshold, slope_threshold_1, slope_threshold_2):
    group_list = []
    while df.shape[0] != 0:
        group = create_group(df, bounding_box_y_range,
                             delta_x_threshold, slope_threshold_1, slope_threshold_2)
        group_list.append(group)
        df = df[~df.label.isin(group)]
    return group_list


def draw_line(group_list, cropped_img, cur_df_centroid, col='y'):
    for group_labels in group_list:
        # find group centrods
        group_centroid = cur_df_centroid[cur_df_centroid.label.isin(
            group_labels)].sort_values(by='y' if col == 'x' else 'x')
        group_centroid['centroid'] = group_centroid.apply(
            lambda df: (df['x'], df['y']), axis=1)
        # draw the line
        c = group_centroid.centroid.values
        line_thickness = 3
        for i in range(0, c.size-1):
            cv2.line(cropped_img, c[i], c[i+1],
                     (0, 255, 255), thickness=line_thickness)
    return cropped_img


def run_plot_whole(out_img, df_building, dict_centroid, boxes, predicted_image, building_index, delta_x_threshold, slope_threshold_1, slope_threshold_2, col='y'):
    cropped_img, bounding_box_min, bounding_box_y_range = crop_image(
        boxes, building_index,predicted_image)
    cur_df_centroid = get_building_centroid(
        df_building, building_index, dict_centroid)
    group_list = find_group_list(cur_df_centroid, bounding_box_y_range,
                                 delta_x_threshold, slope_threshold_1, slope_threshold_2)
    final_image = draw_line(group_list, out_img, cur_df_centroid, col='y')
    list_floor_centroid = find_floor_centroid(group_list, cur_df_centroid)
    num_floor = len(list_floor_centroid)
    # print(f'There are {num_floor} floors are found')
    return final_image, list_floor_centroid, num_floor

def find_floor_centroid(group_list, cur_df_centroid):
    list_floor_centroid = []
    for group in group_list:
        if len(group) < 2:
            continue
        else:
            df_centroid = cur_df_centroid[cur_df_centroid.label.isin(group)]
            list_floor_centroid.append(np.around(df_centroid.y.mean(),2))
    return sorted(list_floor_centroid)


def find_roof_bottom_points(origin_image, building_index, masks, boxes, range_threshold):

    # get contour
    mask_image = masks[building_index]
    res = get_image_info(origin_image, mask_image)
    cnt = res[-1]

    # get bounding box range
    cur_box = boxes.astype('int')[building_index]
    x1, y1, x2, y2 = cur_box[0], cur_box[1], cur_box[2], cur_box[3]
    x_left = int(x1+(1-range_threshold)/2*(x2-x1))
    x_right = int(x2-(1-range_threshold)/2*(x2-x1))
    y_mid = (y2-y1)/2+y1

    # filter out roof and bottom points
    p = cnt.squeeze()
    df_p = pd.DataFrame(p, columns=['x', 'y'])
    df_p_range = df_p[(df_p.x >= x_left) & (df_p.x <= x_right)]
    df_roof = df_p_range[df_p_range.y < y_mid]
    df_bottom = df_p_range[df_p_range.y >= y_mid]

    return df_roof, df_bottom


def linear_simulate(df, show_figure=False):
    X_train = df.x.values.reshape(-1, 1)
    y_train = df.y.values.reshape(-1, 1)

    # Model
    regr = linear_model.LinearRegression()  # Create linear regression object
    regr.fit(X_train, y_train)  # Train the model
    y_pred = regr.predict(X_train)  # Make predictions

    # Evaluation
    coeff = regr.coef_[0][0]  # The coefficients
    mse = mean_squared_error(y_train, y_pred)  # The mean squared error
    # The coefficient of determination: 1 is perfect prediction
    r2 = r2_score(y_train, y_pred)

    if show_figure:
        print('Coefficients(slope): %.3f' % coeff)
        print('Mean squared error: %.2f' % mse)
        print('Evaluation r2 score: %.2f' % r2)

        # Plot outputs
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, 8, color='red')
        plt.plot(X_train, y_pred, color='blue', linewidth=2)
        plt.title("Linear regression")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return coeff, r2


def find_roof_slope(origin_image, building_index, masks, boxes, range_threshold=0.6, r2_socre_threshold=0.8, show_figure=False):

    df_roof, df_bottom = find_roof_bottom_points(
        origin_image, building_index, masks, boxes, range_threshold)
    slope_roof, score_roof = linear_simulate(df_roof, show_figure)
    if score_roof >= r2_socre_threshold:
        slope, score = slope_roof, score_roof
        degree = degrees(atan(-slope))
    else:
        degree, slope, score = np.nan, np.nan, np.nan
    return degree, -slope, score,


def handle_slope_threshold(origin_image, building_index,  masks, boxes, range_threshold, r2_socre_threshold, degree_close=25, degree_far=15, flag=False):
    res = find_roof_slope(origin_image, building_index,  masks, boxes, range_threshold,
                          r2_socre_threshold, show_figure=False)
    slope_degree = res[0]
    s_degree = slope_degree*2/5
    # print(s_degree)
    if not np.isnan(slope_degree) and flag == True:
        slope_sign = np.sign(slope_degree)
        slope_threshold_1 = [tan(radians(s_degree-degree_close+slope_sign*5)),
                             tan(radians(s_degree+degree_close+slope_sign*5))]
        slope_threshold_2 = [tan(radians(s_degree-degree_far+slope_sign*5)),
                             tan(radians(s_degree+degree_far+slope_sign*5))]
    else:
        slope_threshold_1 = [tan(radians(-degree_close)),
                             tan(radians(degree_close))]
        slope_threshold_2 = [tan(radians(-degree_far)),
                             tan(radians(degree_far))]
    return slope_threshold_1, slope_threshold_2
