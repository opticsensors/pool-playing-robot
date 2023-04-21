from pool.cam import Camera
import time
import cv2
from pool.eye import Eye
import numpy as np
from pool.stepper import Stepper
import pandas as pd
import time

def generate_grid(num_horizontal_points, num_vertical_points):
    alpha = 1/(num_horizontal_points+1)
    beta = 1/(num_vertical_points+1)
    points = np.mgrid[1:num_horizontal_points+1,1:num_vertical_points+1].T.reshape(-1,2)
    points = points.astype(np.float32)
    rescaled_points = np.zeros_like(points)
    rescaled_points[:,0] = alpha*points[:,0]
    rescaled_points[:,1] = beta*points[:,1]
    return rescaled_points

def cm_to_steps(incr_x, incr_y, W, H):
    scaler=0.00794
    phi1 = 1/(2*scaler) * (-incr_x*W+incr_y*H)
    phi2 = 1/(2*scaler) * (incr_x*W+incr_y*H)
    return phi1,phi2


##############################################
eye=Eye()

###############################################

cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')
img=cv2.imread(f'./results/corners.jpg')
undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
dist_corners=eye.get_pool_corners(img, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])
undist_corners=eye.get_pool_corners(undistorted, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])

#points=generate_grid(2,2)
#np.random.shuffle(points) 
points=np.array([[    0        ,   0      ],
                 [    0.66667  ,   0.66667],
                 [    0.66667  ,   0.33333],
                 [    0.33333  ,   0.33333]])
print(points)
mode=0
count=0
dict_to_save={}
list_of_dict=[]
pos1 = 0
pos2 = 0
prev_point_x=0
prev_point_y=0
W = 84
H = 44
aruco_to_track=23

for name in ['img_0', 'img_1', 'img_2', 'img_3']:
    
    img = cv2.imread(f'./results/{name}.jpg')
    undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
    dist_warp,dist_matrix=eye.perspective_transform(img,dist_corners)
    undist_warp,undist_matrix=eye.perspective_transform(undistorted,undist_corners)

    # aruco centroid
    x,y=eye.get_aruco_coordinates(img, aruco_to_track)
    cv2.circle(img,(int(x),int(y)), 13, (0,0,255), -1)
    undist_x,undist_y=eye.get_aruco_coordinates(undistorted, aruco_to_track)
    cv2.circle(undistorted,(int(x),int(y)), 13, (0,0,255), -1)
    point_to_transform = np.array([[x,y]], dtype='float32')
    point_to_transform = np.array([point_to_transform])

    transformed_point = cv2.perspectiveTransform(point_to_transform, dist_matrix)
    dist_warp_x,dist_warp_y = transformed_point[0][0]
    cv2.circle(dist_warp,(int(dist_warp_x),int(dist_warp_y)), 13, (0,0,255), -1)
    transformed_point = cv2.perspectiveTransform(point_to_transform, undist_matrix)
    undist_warp_x,undist_warp_y = transformed_point[0][0]
    cv2.circle(undist_warp,(int(undist_warp_x),int(undist_warp_y)), 13, (0,0,255), -1)
    
    cv2.imwrite(f'./results/dist_{name}.jpg', img)
    cv2.imwrite(f'./results/undist_{name}.jpg', undistorted)
    cv2.imwrite(f'./results/dist_warp_{name}.jpg', dist_warp)
    cv2.imwrite(f'./results/undist_warp_{name}.jpg', undist_warp)
    print('error pix coord: ', (x, y), (undist_x, undist_y), (dist_warp_x, dist_warp_y), (undist_warp_x, undist_warp_y))

    point=points[count,:]
    new_point_x,new_point_y=point
    incr_x=new_point_x-prev_point_x
    incr_y=new_point_y-prev_point_y
    pos1,pos2=cm_to_steps(incr_x,incr_y,W,H)
    print('steps send to arduino', pos1, pos2)

    count+=1
    prev_point_x=new_point_x
    prev_point_y=new_point_y