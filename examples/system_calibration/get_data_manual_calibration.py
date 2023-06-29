import cv2
import pandas as pd
from pool.calibration import DataExtractor

img_corners=cv2.imread('./results/corners_0.jpg')
de = DataExtractor(img_corners)

#define needed variables
target_arucos_to_img_number={
    8: [15, 16, 17, 18, 19, 20, 21, 22],
    9: [23, 24, 25, 26, 27, 28, 29, 30],
    10:[69, 70, 71, 72, 73, 74, 75, 76],
    11:[104, 105, 106, 107, 108, 109],
    12:[98, 99, 100, 101, 102, 103],
    13:[1, 2, 3, 4, 5, 6, 7],
    14:[8, 9, 10, 11, 12, 13, 14],
    15:[85, 86, 87, 88, 89, 90, 91],
    16:[60, 61, 62, 63, 64, 65, 66, 67, 68],
    17:[38, 39, 40, 41, 42, 43],
    18:[92, 93, 94, 95, 96, 97],
    19:[54, 55, 56, 57, 58, 59],
    20:[44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    21:[31, 32, 33, 34, 35, 37], #img 36 has to be removed because aruco is out of FOV
    23:[77, 78, 79, 80, 81, 82, 83, 84],
}

flipper_arucos=[76, 77, 78, 79, 80, 81, 82, 83]
carriage_aruco=22

dict_to_save = {}
list_of_dict = []

target_arucos=target_arucos_to_img_number.keys()
d_targets=de.get_several_aruco_data(img_corners, target_arucos)

# get data for every image
for target_aruco, img_numbers in target_arucos_to_img_number.items():
    for img_number in img_numbers:
        img_name=f'img_{img_number}'
        full_img_name=f'img_{img_number}.jpg'
        print(full_img_name)

        img = cv2.imread(f'./results/{full_img_name}')
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

        dict_to_save['extra']=d_coord_flipper['flipper_undist']

        list_of_dict.append(dict_to_save.copy())

# for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df=df.sort_values('img_num')
a=df['extra'].str[0]
df['aruco_flipper_76_x']=a.str[0]
df['aruco_flipper_76_y']=a.str[1]
df['aruco_flipper_77_x']=a.str[2]
df['aruco_flipper_77_y']=a.str[3]

df = df.drop('extra', axis=1)
df=df.dropna() #remove images where at least 2 flipper arucos are not detected
df.to_csv(path_or_buf='./results/calibration_image_data.csv', sep=',',index=False)