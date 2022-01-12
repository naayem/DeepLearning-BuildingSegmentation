import numpy as np
import argparse
import os
import shutil
import time
import cv2
import json
import warnings
import pandas as pd

from glob import glob
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from affnet_util import *

warnings.filterwarnings("ignore")


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def local_kpt_descriptor(img, weight_path, NFEATS=5000, dev='cpu'):
    '''
        use DoG Affnet to extrct the keypoint and descriptor
    '''
    keypoint, descriptor, As = detect_DoG_AffNet_OriNet_HardNet(
        img, weight_path, NFEATS, dev)
    return keypoint, descriptor


def get_descriptors(img_path, weight_path):
    '''
        save the keypoints and descriptors for each image
    '''

    nq = len(img_path)
    print('======== get_descriptors ======== ')
    print('number of images: ', nq)
    list_descriptors = []
    list_keypoints = []
    for i in tqdm(range(nq)):
        img = cv2.imread(img_path[i], cv2.COLOR_BGR2RGB)
        # img = img.astype('uint8')
        keypoint, descriptor = local_kpt_descriptor(img, weight_path)
        list_keypoints.append(keypoint)
        list_descriptors.append(descriptor)
    return list_keypoints, list_descriptors


def affine_inliers(kpts1, kpts2, descs1, descs2):
    '''
        use ratio test and ransav method to delete some mismatched keypoint,
        then get the matched keyoint between two images
    '''
    tentatives = match_snn(descs1, descs2, snn_th=0.85)
    H, inl_mask, inliers_num = ransac_validate(tentatives, kpts1, kpts2, 2.0)
    return inliers_num


def get_inliers(img_path, list_keypoints, list_descriptors):
    '''
        get the matching matrix between every two image
    '''
    print('======== get_inliers ======== ')
    nq = len(img_path)
    inliers = np.zeros([nq, nq])
    num_error = 0
    list_error = []

    for i in tqdm(range(nq-1)):
        kpts1 = list_keypoints[i]
        descs1 = list_descriptors[i]

        for j in range(i+1, nq):
            # print('*',j)
            kpts2 = list_keypoints[j]
            descs2 = list_descriptors[j]
            try:
                inliers_num = affine_inliers(kpts1, kpts2, descs1, descs2)
                # inliers_num_1 = affine_inliers(kpts1,kpts2,descs1,descs2)
                # if inliers_num_1 >0:
                #   inliers_num_2 = affine_inliers(kpts1,kpts2,descs1,descs2)
                # inliers_num = np.min([inliers_num_1,inliers_num_2])
            except:
                inliers_num = 0
                list_error.append([i, j])
                num_error += 1
            inliers[i][j] = inliers_num
    return inliers


def distance_matix(X):
    '''
        convert the matching matrix to distance matrix
    '''
    X = X + X.T - np.diag(np.diag(X))
    X = 1/X
    X[np.diag_indices_from(X)] = 0
    X = np.nan_to_num(X, copy=True, nan=0, posinf=1, neginf=-1)
    return X


def dbscan(distance_matrix, eps_range, min_samples=4):
    '''
        use dbscan to do the image clustring based on the distance matrix
    '''
    eps_diff = eps_range[1]-eps_range[0]
    eps_final = -1
    max_num_cluster = -1
    
    print(f'======== DBSCAN clstering ==========')
    print('cluster -1 is a noisy cluster')
    # loop the eps from (0.1,0.6,0.02) to find the result have the largest number of clusters
    for eps in eps_range:
        clustering = DBSCAN(eps, min_samples).fit(distance_matrix)
        c = clustering.labels_
        label_unique = np.unique(c)
        print(f'eps:{eps}  clusters:{label_unique} ')
        num_cluster = len(label_unique)

        if eps == eps_range[0]:
            max_num_cluster = num_cluster
        else:
            if max_num_cluster <= num_cluster:
                max_num_cluster = num_cluster
            else:
                eps_final = eps - eps_diff
                break

    print(f'used eps is : {eps_final}')
    if eps_final<0:
        eps_final = eps_range[-1]
    clustering_res = DBSCAN(eps_final, min_samples).fit(distance_matrix)
    cluster_labels = clustering_res.labels_

    return cluster_labels


def save_clustered_img(label_list, img_path, cluster_address):
    '''
        save the clustered builidng duplication 
    '''
    print(f'======== Result ==========')
    set_label = set(label_list)
    noise_flag = (-1 in set_label)
    print(f'There are {len(set_label)-(noise_flag)} clusters')
    for label in set_label:
        ind = np.where(label_list == label)[0]
        list_img = []
        for i in ind:
            list_img.append(img_path[i])
        cluster_path = f'{cluster_address}/{label}'
        make_dir(cluster_path)
        for path in list_img:
            img_name = path.split('/')[-1]
            out_path = f'{cluster_path}/{img_name}'
            shutil.copy(path, cluster_path)


def duplication_clustering(eps_range, min_samples, id_segment, dataset_dir, clustered_img_dir, matching_matrix_dir, weight_path):
    '''
        main function to run the duplication detection
    '''
    start_time = time.time()
    print(f'========  Segment id: {id_segment}  ========')

    clustered_segment_dir = f'{clustered_img_dir}/{id_segment}'
    make_dir(clustered_segment_dir)

    img_path = sorted(glob(os.path.join(dataset_dir, "*.jpg")))
    nq = len(img_path)

    # matching matrix
    matrix_path = f'{matching_matrix_dir}/{id_segment}.npy'
    if os.path.exists(matrix_path):
        print('use existed matching matrix')
        inliers_matrix = np.load(matrix_path)
    else:
        list_keypoints, list_descriptors = get_descriptors(
            img_path, weight_path)
        inliers_matrix = get_inliers(
            img_path, list_keypoints, list_descriptors)
        if inliers_matrix.shape[0] != 0:
            np.save(matrix_path, inliers_matrix)

    # convert to distance matrix
    distance_matrix = distance_matix(inliers_matrix)

    label_list = dbscan(distance_matrix, eps_range, min_samples)
    save_clustered_img(label_list, img_path, clustered_segment_dir)
    print("======== running time: %.3s seconds ========" %
          (time.time() - start_time))


def distance(df, model1_centroid):
    '''
        calculate the distance between two builidng centroid coordinate
    '''
    if pd.isna(df.building_centroid_coordinate):
        return np.nan
    model2_centroid = json.loads(df.building_centroid_coordinate)
    dist = np.sqrt(model2_centroid[1]-model1_centroid[1]
                   )**2+(model2_centroid[0]-model1_centroid[0])**2
    return dist


def find_building_type(building_index, model1_csv_path, model2_csv_path, distance_threshold=400):
    '''
        link the builidng type of model1 and model2 builidng instance by checking the distance 
    '''
    df_model1 = pd.read_csv(model1_csv_path, index_col=0)
    df_model2 = pd.read_csv(model2_csv_path, index_col=0)

    unique = df_model2['building_type'].unique()
    if len(unique) == 1:
        if pd.isnull(unique):
            return np.nan
        return unique[0]

    model1_centroid = df_model1[df_model1['building_index'] == int(
        building_index)].building_centroid_coordinate.values[0]
    model1_centroid = json.loads(model1_centroid)
    df_model2['distance'] = df_model2.apply(
        lambda x: distance(x, model1_centroid), axis=1)
    df_building_type = df_model2[df_model2['distance']
                                 <= distance_threshold].sort_values('distance')
    if df_building_type.size != 0:
        building_type = df_building_type.loc[:, 'building_type'].iloc[0]
    else:
        building_type = np.nan
    return building_type


def combine_builidng_type(id_segment, clustered_img_dir, clustered_building_type_dir, model1_open_info_folder, model2_building_info_folder):
    '''
        main funciton to combine the builidng type
    '''
    dataset = sorted(glob(f'{clustered_img_dir}/{id_segment}/*'))
    df_cluster_building_type = pd.DataFrame(
        columns=['cluster_name', 'building_type', 'id_segment'])
    list_cluster_name = []  # record cluster builidng type
    list_cluster_type = []  # record each instance's builidng type in the cluster

    for cluster_folder in dataset:
        cluster_name = cluster_folder.split('/')[-1]
        if cluster_name != '-1':  # filter out the noisy cluster name '-1'
            list_cluster_name.append(cluster_name)
            img_paths = sorted(glob(os.path.join(cluster_folder, "*.jpg")))
            list_imgs_type = []
            for img_path in img_paths:

                # building image's info(frame, name). use this info to link the result of model1 and model2
                crop_img_name = img_path.split('/')[-1]
                building_index = crop_img_name.split('.')[0].split('_')[-1]
                original_img_name = crop_img_name.split(
                    '.')[0][:-len(building_index)-1]
                crop_csv_name = original_img_name + '.csv'
                type_csv_name = original_img_name + '.csv'
                model1_csv_path = f'{model1_open_info_folder}/{id_segment}/{crop_csv_name}'
                model2_csv_path = f'{model2_building_info_folder}/{id_segment}/{type_csv_name}'
                # find building type for each image in the cluster
                building_type = find_building_type(
                    building_index, model1_csv_path, model2_csv_path, distance_threshold=400)
                if pd.notnull(building_type):
                    list_imgs_type.append(building_type)
            # find the most common builidng type in the cluster
            cluster_type = max(list_imgs_type, key=list_imgs_type.count)
            list_cluster_type.append(cluster_type)

    df_cluster_building_type['cluster_name'] = list_cluster_name
    df_cluster_building_type['building_type'] = list_cluster_type
    df_cluster_building_type['id_segment'] = id_segment

    save_dir = f'{clustered_building_type_dir}/{id_segment}'
    make_dir(save_dir)
    save_path = f'{save_dir}/{id_segment}_building_type.csv'
    df_cluster_building_type.to_csv(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--weight_path', default='/path/to/affnet_weight',
                        type=str, help="affnet model pretrained weight")
    parser.add_argument('--segment_cropped_image_path', default='/path/to/segments', type=str,
                                                                help="cropped building instance images after using model 1")
    parser.add_argument('--id_segments', default='16888', type=str,
                        help="segments' id for testing(delimited list input)")
    parser.add_argument('--result_path', default='/path/to/result', type=str,
                        help="this folder is used to save the clustered builidng duplication and matching matrix between the instances")
    parser.add_argument('--model1_open_info_folder', default='./result/model1/open_info',
                        type=str, help="centroid coordinate of building instance predicted by model1")
    parser.add_argument('--model2_building_info_folder', default='./result/model2/building_info',
                        type=str, help="building type info predicted by model2")
    args = parser.parse_args()

    print('============  Arguments infos ============ ')
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))

    # ============ run dupliction detection to cluster building  ============
    SEGMENTS_FOLDER = args.segment_cropped_image_path
    RESULT_FOLDER = args.result_path
    # make_dir(f'./result')
    make_dir(f'{RESULT_FOLDER}')

    clustered_img_dir = f'{RESULT_FOLDER}/clustered_imgs'
    matching_matrix_dir = f'{RESULT_FOLDER}/matching_matrix'
    clustered_building_type_dir = f'{RESULT_FOLDER}/cluster_building_type'
    make_dir(clustered_img_dir)
    make_dir(matching_matrix_dir)
    make_dir(clustered_building_type_dir)

    test_segments = [item for item in args.id_segments.split(',')]
    eps_range = np.arange(0.1, 0.6, 0.02)
    min_samples = 4

    for id_segment in test_segments:
        dataset_dir = f'{SEGMENTS_FOLDER}/{str(id_segment)}'
        duplication_clustering(eps_range, min_samples, id_segment, dataset_dir,
                               clustered_img_dir, matching_matrix_dir, args.weight_path)
        combine_builidng_type(id_segment, clustered_img_dir, clustered_building_type_dir,
                              args.model1_open_info_folder, args.model2_building_info_folder)

    print(f'============ Finish, result is saved at {args.result_path}  ============ ')
