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
dist_matrix=eye.calculate_perspective_matrix(dist_corners)
undist_matrix=eye.calculate_perspective_matrix(undist_corners)

#define needed variables
points=np.load('./data/calibration_points.npy')
home_point = [0,0]
points = np.vstack([home_point,points]) # img_0 is when end effector is at home
h_num_points=np.unique(points[:,0]).shape[0]
v_num_points=np.unique(points[:,1]).shape[0]
d_points={}
for i,point in enumerate(points):
    d_points[i]=point
dict_to_save = {}
list_of_dict = []
W = 70.25
H = 38.5
aruco_to_track=22

# get data for every image
for full_img_name in os.listdir('./stepper_calibration/'):

    img_name=os.path.splitext(full_img_name)[0]
    img_format=os.path.splitext(full_img_name)[1]

    if img_format=='.jpg':
        print(full_img_name)
        img_number=int(img_name.split('_')[1])
        img = cv2.imread(f'./stepper_calibration/{full_img_name}')
        undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)

        x_dist,y_dist=eye.get_aruco_coordinates(img, aruco_to_track)
        x_undist,y_undist=eye.get_aruco_coordinates(undistorted, aruco_to_track)

        point_to_transform = np.array([[x_dist,y_dist]], dtype='float32')
        point_to_transform = np.array([point_to_transform])
        transformed_point = cv2.perspectiveTransform(point_to_transform, dist_matrix)
        x_dist_warp, y_dist_warp = transformed_point[0][0]

        point_to_transform = np.array([[x_undist,y_undist]], dtype='float32')
        point_to_transform = np.array([point_to_transform])
        transformed_point = cv2.perspectiveTransform(point_to_transform, undist_matrix)
        x_undist_warp, y_undist_warp = transformed_point[0][0]
        dict_to_save['img_num']=img_number
        dict_to_save['img_name']=img_name
        dict_to_save['x_dist']=x_dist
        dict_to_save['y_dist']=y_dist
        dict_to_save['x_undist']=x_undist
        dict_to_save['y_undist']=y_undist
        dict_to_save['x_dist_warp']=x_dist_warp
        dict_to_save['y_dist_warp']=y_dist_warp    
        dict_to_save['x_undist_warp']=x_undist_warp
        dict_to_save['y_undist_warp']=y_undist_warp             
        dict_to_save['point_x']=d_points[img_number][0]            
        dict_to_save['point_y']=d_points[img_number][1]  

        list_of_dict.append(dict_to_save.copy())

# for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df=df.sort_values('img_num')
df.to_csv(path_or_buf=f'./data/calibration_image_data.csv', sep=' ',index=False)

incr_df=df.drop(columns=['img_num', 'img_name'])
incr_df = incr_df.rename(columns={'x_dist':        'incr_x_dist', 
                                  'y_dist':        'incr_y_dist',
                                  'x_undist':      'incr_x_undist',
                                  'y_undist':      'incr_y_undist',
                                  'x_dist_warp':   'incr_x_dist_warp',
                                  'y_dist_warp':   'incr_y_dist_warp',
                                  'x_undist_warp': 'incr_x_undist_warp',
                                  'y_undist_warp': 'incr_y_undist_warp',
                                  'point_x':       'incr_point_x',
                                  'point_y':       'incr_point_y'})   
incr_df=incr_df.diff()

#functions to create new dataframe columns
def cm_to_steps1(incr_x, incr_y, W, H):
    scaler=0.00794
    phi1 = 1/(2*scaler) * (-incr_x*W+incr_y*H)
    return phi1

def cm_to_steps2(incr_x, incr_y, W, H):
    scaler=0.00794
    phi2 = 1/(2*scaler) * (incr_x*W+incr_y*H)
    return phi2

def img_num_to_incr_id(img_num):
    if img_num==0:
        return np.nan
    else:
        return f'{img_num-1}-{img_num}'

incr_df['incr_id'] = df.apply(lambda x: img_num_to_incr_id(x['img_num']), axis=1)
incr_df['incr_steps1'] = incr_df.apply(lambda x: cm_to_steps1(x['incr_point_x'], x['incr_point_y'],W,H), axis=1)
incr_df['incr_steps2'] = incr_df.apply(lambda x: cm_to_steps2(x['incr_point_x'], x['incr_point_y'],W,H), axis=1)
incr_df=incr_df.dropna()
incr_df['incr_steps1_int'] = (incr_df['incr_steps1']).astype(int)
incr_df['incr_steps2_int'] = (incr_df['incr_steps2']).astype(int)
incr_df.to_csv(path_or_buf=f'./data/calibration_pixel_to_step.csv', sep=' ',index=False)

