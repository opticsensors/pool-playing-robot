import cv2
import pandas as pd
import os 
from pool.calibration import DataExtractor, InverseKinematics

#helper functions for calibration
ik = InverseKinematics()

img_corners=cv2.imread('./results/data2/img_1.jpg')
de = DataExtractor(img_corners)

#define needed variables
points=pd.read_csv('./results/data2/calibration_points.csv', sep=',',decimal='.')

dict_to_save = {}
list_of_dict = []
aruco_to_track = 22

# get data for every image
for full_img_name in os.listdir('./results/data2/'):

    img_name=os.path.splitext(full_img_name)[0]
    img_format=os.path.splitext(full_img_name)[1]

    if img_format=='.jpg':
        print(full_img_name)
        img_number=int(img_name.split('_')[1])
        img = cv2.imread(f'./results/data2/{full_img_name}')
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

        list_of_dict.append(dict_to_save.copy())

# for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df=df.sort_values('img_num')
df.to_csv(path_or_buf='./results/data2/calibration_image_data.csv', sep=',',index=False)

df_merged=pd.merge(df, points, on='img_num')

incr_df=df.drop(columns=['img_num', 'img_name'])    
columns_to_diff=['x_dist','y_dist','x_undist', 'y_undist', 'x_dist_warp','y_dist_warp','x_undist_warp','y_undist_warp']
columns_to_save=['incr_x_dist', 'incr_y_dist','incr_x_undist','incr_y_undist','incr_x_dist_warp','incr_y_dist_warp','incr_x_undist_warp','incr_y_undist_warp',]
df_merged[columns_to_save]=df_merged[columns_to_diff].diff()

df_merged['incr_id'] = df_merged.apply(lambda x: ik.img_num_to_incr_id(x['img_num']), axis=1) #info about the images where the increment took place
df_merged=df_merged.iloc[1:] # drop first row 
df_merged.to_csv(path_or_buf=f'./results/data2/calibration_pixel_to_step.csv', sep=',',index=False)