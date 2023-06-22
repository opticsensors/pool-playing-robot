import cv2
from pool.eye import Eye
import numpy as np
import time

eye=Eye()
cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')
img=cv2.imread('./data/corners_0.jpg')
undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
corners=eye.get_pool_corners(undistorted, bottom_aruco_ids=[6,7],top_aruco_ids=[2,3],left_aruco_ids=[0,1],right_aruco_ids=[4,5])
print(corners)

#name='img_0'
#img = cv2.imread(f'./results/{name}.jpg')
#undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
warp,_=eye.perspective_transform(undistorted,corners)

#x,y=eye.get_aruco_coordinates(img, 23)
x,y=eye.get_aruco_coordinates(warp, 23)

#cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
cv2.circle(warp, (int(x), int(y)), 6, (0, 0, 255), -1)

#cv2.imshow('', cv2.resize(img,(0,0),fx=0.25,fy=0.25))
cv2.imshow('', cv2.resize(warp,(0,0),fx=0.25,fy=0.25))

cv2.waitKey(0)