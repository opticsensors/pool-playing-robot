import cv2
from pool.eye import Eye
import numpy as np
import pandas as pd
import os 
from pool import corners
from pool.pixel_to_steps import PixelSteps

#CV algorithms initialization
eye=Eye()

#helper functions for calibration
ps = PixelSteps()

#compute corners 
img=corners.load_img_corners()
undist_img=eye.undistort_image(img, remapping=False)
dist_corners=eye.get_pool_corners(img)
undist_corners=eye.get_pool_corners(undist_img)

#compute prespective transform matrix
# the prespective trans matrix should be the same for all the captured images
dist_matrix=eye.calculate_perspective_matrix(dist_corners)
undist_matrix=eye.calculate_perspective_matrix(undist_corners)

#define needed variables
points=ps.load_calibration_points()
points=ps.add_homing_position(points) #first image is in home pos
d_points=ps.points_to_dict(points)
dict_to_save = {}
list_of_dict = []
aruco_to_track=22

# get data for every image
for full_img_name in os.listdir('./results/'):

    img_name=os.path.splitext(full_img_name)[0]
    img_format=os.path.splitext(full_img_name)[1]

    if img_format=='.jpg':
        print(full_img_name)
        img_number=int(img_name.split('_')[1])
        img = cv2.imread(f'./results/{full_img_name}')
        undistorted=eye.undistort_image(img, remapping=False)

        x_dist,y_dist=eye.get_aruco_coordinates(img, aruco_to_track)
        x_undist,y_undist=eye.get_aruco_coordinates(undistorted, aruco_to_track)

        x_dist_warp, y_dist_warp = eye.transform_point_given_a_matrix((x_dist,y_dist),dist_matrix)
        x_undist_warp, y_undist_warp = eye.transform_point_given_a_matrix((x_undist,y_undist),undist_matrix)

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
df.to_csv(path_or_buf='./results/calibration_image_data.csv', sep=',',index=False)

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

incr_df['incr_id'] = df.apply(lambda x: ps.img_num_to_incr_id(x['img_num']), axis=1)
incr_df=incr_df.dropna()
incr_x = incr_df['incr_point_x'].values
incr_y = incr_df['incr_point_y'].values
incr_steps1, incr_steps2 = ps.cm_to_steps_vectorized(incr_x, incr_y)
incr_df['incr_steps1']=incr_steps1.astype(int)
incr_df['incr_steps2']=incr_steps2.astype(int)
incr_df.to_csv(path_or_buf=f'./results/calibration_pixel_to_step.csv', sep=',',index=False)

