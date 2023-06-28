import cv2
import pandas as pd
import os 
from pool.calibration import DataExtractor, PixelSteps

#helper functions for calibration
ps = PixelSteps()

img_corners=cv2.imread('./results/corners_0.jpg')
de = DataExtractor(img_corners)

#define needed variables
points=ps.load_calibration_points()
points=ps.add_homing_position(points) #first image is in home pos
d_points=ps.points_to_dict(points)
dict_to_save = {}
list_of_dict = []
aruco_to_track = 22

# get data for every image
for full_img_name in os.listdir('./results/'):

    img_name=os.path.splitext(full_img_name)[0]
    img_format=os.path.splitext(full_img_name)[1]

    if img_format=='.jpg':
        print(full_img_name)
        img_number=int(img_name.split('_')[1])
        img = cv2.imread(f'./results/{full_img_name}')
        d_aruco=de.get_single_aruco_data(img, aruco_to_track)

        dict_to_save['img_num']=img_number
        dict_to_save['img_name']=img_name
        dict_to_save['x_dist']=d_aruco['dist'][0]
        dict_to_save['y_dist']=d_aruco['dist'][1]
        dict_to_save['x_undist']=d_aruco['undist'][0]
        dict_to_save['y_undist']=d_aruco['undist'][1]
        dict_to_save['x_dist_warp']=d_aruco['dist_warp'][0]
        dict_to_save['y_dist_warp']=d_aruco['dist_warp'][1]    
        dict_to_save['x_undist_warp']=d_aruco['undist_warp'][0]
        dict_to_save['y_undist_warp']=d_aruco['undist_warp'][1]           
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
incr_steps1, incr_steps2 = ps.cm_to_steps(incr_x, incr_y)
incr_df['incr_steps1']=incr_steps1.astype(int)
incr_df['incr_steps2']=incr_steps2.astype(int)
incr_df.to_csv(path_or_buf=f'./results/calibration_pixel_to_step.csv', sep=',',index=False)

