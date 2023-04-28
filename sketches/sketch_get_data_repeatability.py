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
img=cv2.imread(f'./data/corners.jpg')
undist_img=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
dist_corners=eye.get_pool_corners(img, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])
undist_corners=eye.get_pool_corners(undist_img, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])

#compute prespective transform matrix
# the prespective trans matrix should be the same for all the captured images
_,dist_matrix=eye.perspective_transform(img,dist_corners)
_,undist_matrix=eye.perspective_transform(undist_img,undist_corners)

#define needed variables
points=np.load('./data/repeatability_points.npy')
#unique_points=np.unique(points, axis=0)
home_point = [0,0]
points = np.vstack([home_point,points]) # img_0 is when end effector is at home
#h_num_points=np.unique(points[:,0]).shape[0]
#v_num_points=np.unique(points[:,1]).shape[0]
#num_points_without_repeatability=h_num_points*v_num_points
#repeatability=points.shape[0]//num_points_without_repeatability
d_points={}
for i,point in enumerate(points):
    d_points[i]=point
dict_to_save = {}
list_of_dict = []
W = 84
H = 44
aruco_to_track=23

# get data for every image
for full_img_name in os.listdir('./stepper_repeatability/'):

    img_name=os.path.splitext(full_img_name)[0]
    img_format=os.path.splitext(full_img_name)[1]

    if img_format=='.jpg':
        print(full_img_name)
        img_number=int(img_name.split('_')[1])
        img = cv2.imread(f'./stepper_repeatability/{full_img_name}')
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
df.to_csv(path_or_buf=f'./data/repeatability_image_data.csv', sep=' ',index=False)

#get unique points
#for point in unique_points:
#    point_x=point[0]
#    point_y=point[1]
#    same_point_df = df.loc[(df['point_x'] == point_x) & (df['point_y'] == point_y)]
#    x_undist_warp = same_point_df['x_undist_warp']
#    y_undist_warp = same_point_df['y_undist_warp']
#    error= ((df.p - df.x) ** 2).mean() ** .5


df['point_id'] = df.groupby(['point_x','point_y']).ngroup()

def std(x): 
    return np.std(x)

#rms_x_undist_warp = df.groupby(['point_x', 'point_y'])["x_undist_warp"].apply(std)
rms_x_undist_warp = df.groupby(['point_id'])["x_undist_warp"].apply(std)
#rms_y_undist_warp = df.groupby(['point_x', 'point_y'])["y_undist_warp"].apply(std)
rms_y_undist_warp = df.groupby(['point_id'])["y_undist_warp"].apply(std)

out = pd.DataFrame([rms_x_undist_warp, rms_y_undist_warp]).transpose()
out.to_csv(path_or_buf=f'./data/repeatability_error.csv', sep=' ',index=False)
