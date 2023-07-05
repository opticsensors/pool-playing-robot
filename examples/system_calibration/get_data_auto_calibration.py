import cv2
import pandas as pd
from pool.calibration import DataExtractor

img_corners=cv2.imread('./results/auto_data1/img_1.jpg')
img_arucos=cv2.imread('./results/auto_data1/img_arucos.jpg')

de = DataExtractor(img_corners)

#define needed variables
target_arucos_to_img_number={
    13:[5],
    15:[4],
    17:[0],
    18:[3],
    20:[2],
    21:[1],
}

flipper_arucos=[76, 77, 78, 79, 80, 81, 82, 83]
carriage_aruco=22

dict_to_save = {}
list_of_dict = []

target_arucos=target_arucos_to_img_number.keys()
d_targets=de.get_several_aruco_data(img_arucos, target_arucos)

# get data for every image
for target_aruco, img_numbers in target_arucos_to_img_number.items():
    for img_number in img_numbers:
        img_name=f'img_{img_number}'
        full_img_name=f'img_{img_number}.jpg'
        print(full_img_name)

        img = cv2.imread(f'./results/auto_data1/{full_img_name}')
        d_carriage_aruco=de.get_single_aruco_data(img, carriage_aruco)
        d_angle_flipper=de.get_angle_given_line_of_arucos(img, flipper_arucos)
        d_coord_flipper=de.get_coord_given_line_of_arucos(img, flipper_arucos)

        dict_to_save['img_num']=img_number
        dict_to_save['target_aruco']=target_aruco
        dict_to_save['img_name']=img_name

        dict_to_save['carriage_x_dist']=d_carriage_aruco['dist'][0]
        dict_to_save['carriage_y_dist']=d_carriage_aruco['dist'][1]
        dict_to_save['carriage_x_undist']=d_carriage_aruco['undist'][0]
        dict_to_save['carriage_y_undist']=d_carriage_aruco['undist'][1]
        dict_to_save['carriage_x_dist_warp']=d_carriage_aruco['dist_warp'][0]
        dict_to_save['carriage_y_dist_warp']=d_carriage_aruco['dist_warp'][1]    
        dict_to_save['carriage_x_undist_warp']=d_carriage_aruco['undist_warp'][0]
        dict_to_save['carriage_y_undist_warp']=d_carriage_aruco['undist_warp'][1]

        dict_to_save['target_x_dist']=d_targets['d_dist'][target_aruco][0]
        dict_to_save['target_y_dist']=d_targets['d_dist'][target_aruco][1]
        dict_to_save['target_x_undist']=d_targets['d_undist'][target_aruco][0]
        dict_to_save['target_y_undist']=d_targets['d_undist'][target_aruco][1]
        dict_to_save['target_x_dist_warp']=d_targets['d_dist_warp'][target_aruco][0]
        dict_to_save['target_y_dist_warp']=d_targets['d_dist_warp'][target_aruco][1]    
        dict_to_save['target_x_undist_warp']=d_targets['d_undist_warp'][target_aruco][0]
        dict_to_save['target_y_undist_warp']=d_targets['d_undist_warp'][target_aruco][1]

        dict_to_save['angle_dist']=d_angle_flipper['angle_dist']
        dict_to_save['angle_undist']=d_angle_flipper['angle_undist']
        dict_to_save['angle_dist_warp']=d_angle_flipper['angle_dist_warp']
        dict_to_save['angle_undist_warp']=d_angle_flipper['angle_undist_warp']

        list_of_dict.append(dict_to_save.copy())

# for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df=df.sort_values('img_num')
df.to_csv(path_or_buf='./results/auto_data1/calibration_image_data.csv', sep=',',index=False)

points=pd.read_csv(f'./results/auto_data1/calibration_points.csv', sep=',',decimal='.')
df_merged=pd.merge(df, points, on='img_num')
df_merged.to_csv(path_or_buf=f'./results/auto_data1/calibration_dof_to_step.csv', sep=',',index=False)