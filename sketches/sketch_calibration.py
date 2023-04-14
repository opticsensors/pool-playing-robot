import cv2
from pool.eye import Eye
import numpy as np

eye=Eye()
eye.bottom_aruco_ids=[17,8]
eye.top_aruco_ids=[9,3]
eye.left_aruco_ids=[1,11]
eye.right_aruco_ids=[23,10]

#read image with random ball config 
name='config_1'
img = cv2.imread(f'./data/{name}.jpg')
cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')

undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
corners=eye.get_pool_corners(undistorted)
warp=eye.perspective_transform(undistorted,corners)

cv2.imwrite(f'./results/undistorted_{name}.jpg', warp)
cv2.imwrite(f'./results/warp_{name}.jpg', warp)

