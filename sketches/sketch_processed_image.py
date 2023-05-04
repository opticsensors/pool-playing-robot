import cv2
from pool.eye import Eye
import numpy as np
import pandas as pd
import time
import os 

#CV algorithms initialization
eye=Eye()

#load precalibrated data
cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')

#compute corners ---------------------- picture corners.jpg should be updated in another script or in here in case the structure moves
img=cv2.imread(f'./data/corners_0.jpg')
undist_img=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
dist_corners=eye.get_pool_corners(img, bottom_aruco_ids=[6,7],top_aruco_ids=[2,3],left_aruco_ids=[0,1],right_aruco_ids=[4,5])
undist_corners=eye.get_pool_corners(undist_img, bottom_aruco_ids=[6,7],top_aruco_ids=[2,3],left_aruco_ids=[0,1],right_aruco_ids=[4,5])

#compute prespective transform matrix
# the prespective trans matrix should be the same for all the captured images
_,dist_matrix=eye.perspective_transform(img,dist_corners)
_,undist_matrix=eye.perspective_transform(undist_img,undist_corners)

img_name='img_0.jpg'
img = cv2.imread(f'./stepper_calibration/{img_name}')
undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
warp_distorted=eye.perspective_transform(img,dist_corners)
warp_undistorted=eye.perspective_transform(undistorted,undist_corners)

cv2.imwrite('./results/img_undist_warp.jpg', )