import cv2
from pool.eye import Eye
import numpy as np
import time

eye=Eye()
img=cv2.imread(f'./results/corners.jpg')
corners=eye.get_pool_corners(img, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])
cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')


name='img_0'
img = cv2.imread(f'./results/{name}.jpg')
undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
warp=eye.perspective_transform(undistorted,corners)

x,y=eye.get_aruco_coordinates(img, 23)
#x,y=eye.get_aruco_coordinates(warp, 23)

cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
#cv2.circle(warp, (x, y), 4, (0, 0, 255), -1)

cv2.imshow('', cv2.resize(img,(0,0),fx=0.25,fy=0.25))
#cv2.imshow('', warp)
cv2.waitKey(0)