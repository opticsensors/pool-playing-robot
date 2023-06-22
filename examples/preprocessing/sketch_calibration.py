import cv2
from pool.eye import Eye
import numpy as np

eye=Eye()

#read image with random ball config 
name='config_1'
img = cv2.imread(f'./data/{name}.jpg')
cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')

undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
corners=eye.get_pool_corners(undistorted, bottom_aruco_ids=[17,8],top_aruco_ids=[9,3],left_aruco_ids=[1,11],right_aruco_ids=[23,10])
warp=eye.perspective_transform(undistorted,corners)

cv2.imwrite(f'./results/undistorted_{name}.jpg', undistorted)
cv2.imwrite(f'./results/warp_{name}.jpg', warp)

