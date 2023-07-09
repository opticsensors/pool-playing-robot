import cv2
import pandas as pd
import numpy as np
from pool.eye import Eye
from pool.ball_detection import Yolo, ClassicCV

eye=Eye()
yolo=Yolo()
cv=ClassicCV()

dict_to_save = {}
list_of_dict = []

img_corners = cv2.imread('./results/corners_0.jpg')
img_corners_undist=eye.undistort_image(img_corners, remapping=False)
img_corners_undist_warp=eye.undistort_and_warp_image(img_corners,img_corners)
undist_corners=eye.get_pool_corners(img_corners_undist)
undist_matrix=eye.calculate_perspective_matrix(undist_corners)

for img_num in [1,2,3,4,5,6,7,8,10,11,12,13,14]:
    print(img_num)

    # undist warp img
    img=cv2.imread(f'./results/img_{img_num}.jpg')
    img_undist=eye.undistort_image(img, remapping=False)
    img_undist_warp=eye.undistort_and_warp_image(img, img_corners)

    # find centroids using diff methods
    d_centroids_undist, _ = yolo.detect_balls(img_undist,conf=0.25, overlap_threshold=100)
    d_centroids_yolo_img, _ = yolo.detect_balls(img_undist_warp,conf=0.25, overlap_threshold=100)

    d_centroids_yolo_points = {}
    for ball_num,point in d_centroids_undist.items():
        d_centroids_yolo_points[ball_num] = eye.transform_point_given_a_matrix(point, undist_matrix)

    d_centroids_classic_cv=cv.detect_balls(img_undist_warp, img_corners_undist_warp, cloth_color='tuned', color_space='hsv')

    d_real_centroids={}
    for ball_id in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        img_ball=cv2.imread(f'./results/ground_truth_img_{img_num}/{ball_id}.png')
        M = cv2.moments(img_ball[...,0])
        # calculate x,y coordinate of center
        centroid= [(M["m10"] / M["m00"]),(M["m01"] / M["m00"])]
        d_real_centroids[ball_id]=centroid

    dicts = [d_real_centroids,d_centroids_yolo_img,d_centroids_yolo_points,d_centroids_classic_cv]
    common_keys = set(d_real_centroids.keys())
    for d in dicts[1:]:
        common_keys.intersection_update(set(d.keys()))

    common_detected_balls=list(common_keys)

    for ball_num in common_detected_balls:

        centroid_yolo_img = np.array(d_centroids_yolo_img[ball_num])
        centroid_yolo_points = np.array(d_centroids_yolo_points[ball_num])
        centroid_classic = np.array(d_centroids_classic_cv[ball_num])
        centroid_real = np.array(d_real_centroids[ball_num])

        dict_to_save['img_num']=img_num
        dict_to_save['ball_num']=ball_num
        dict_to_save['yolo_img_x'] = centroid_yolo_img[0]
        dict_to_save['yolo_img_y'] = centroid_yolo_img[1]
        dict_to_save['yolo_points_x'] = centroid_yolo_points[0]
        dict_to_save['yolo_points_y'] = centroid_yolo_points[1]
        dict_to_save['classics_x'] = centroid_classic[0]
        dict_to_save['classic_y'] = centroid_classic[1]
        dict_to_save['real_x'] = centroid_real[0]
        dict_to_save['real_y'] = centroid_real[1]
        dict_to_save['error_yolo_img'] = np.linalg.norm(centroid_yolo_img - centroid_real)
        dict_to_save['error_yolo_points'] = np.linalg.norm(centroid_yolo_points - centroid_real)
        dict_to_save['error_classic'] = np.linalg.norm(centroid_classic - centroid_real)

        list_of_dict.append(dict_to_save.copy())

# for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df=df.sort_values('img_num')
df.to_csv(path_or_buf='./results/error_data.csv', sep=',',index=False)