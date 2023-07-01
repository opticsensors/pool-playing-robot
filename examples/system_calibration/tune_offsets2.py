import numpy as np
import cv2
import pandas as pd
from pool.eye import Eye
from pool.calibration import DataExtractor, CameraCalibration

eye=Eye()

img_corners = cv2.imread('./results/corners_0.jpg')
img_corners = eye.undistort_image(img_corners, remapping=False)

calib=CameraCalibration()
de = DataExtractor(img_corners)
rvec,tvec=calib.load_world_frame()
df = pd.read_csv('./results/calibration_image_data.csv', sep=',',decimal='.')

l_valid_offsets=[]

a=26.4/25
b=74.1/25
h=220/25 - 19/25
d=22/25

width_a=0.3
width_b=0.8
width_h=1.1
width_d=0.2
range_a=np.linspace(a-width_a, a+width_a, num=20)
range_b=np.linspace(b-width_b, b+width_b, num=20)
range_h=np.linspace(h-width_h, h+width_h, num=20)
range_d=np.linspace(d-width_d, d+width_d, num=20)

for index, row in df.iterrows():

    target=row[['target_x_undist','target_y_undist']].values
    carriage=row[['carriage_x_undist','carriage_y_undist']].values
    angle=row[['angle_undist']].values[0]

    uvBallCentroid=np.pad(target, (0, 1), constant_values=1).reshape(-1,1)
    trans=calib.transform_image_point_to_world_point(uvBallCentroid, 0, rvec, tvec) #translation vector from world frame to auxiliar frame 

    angle=angle-90 # conversion angle uv frame to angle auxiliar frame (the frame with x axis with the same direction as flipper)
    angle=np.radians(angle)
    Rt=np.array([[np.cos(angle), -np.sin(angle), 0, trans[0]],
                [np.sin(angle), np.cos(angle), 0, trans[1]],
                [0, 0, 1, trans[2]],
                [0, 0, 0, 1]])

    for val_a in range_a:
        for val_b in range_b:
            for val_d in range_d:
                for val_h in range_h:
                    offset_point_of_contact=np.array([-val_b,-val_a,-val_h,1]).reshape(-1,1)
                    offset_rotation_axis=np.array([val_d,0,0,1]).reshape(-1,1)
                    carriage_aruco_point3D = (Rt@offset_point_of_contact)[:3,0] + offset_rotation_axis[:3,0]
                    carriage_aruco_point2D = calib.transform_world_point_to_img_point(carriage_aruco_point3D ,rvec, tvec)
                    error=np.linalg.norm(carriage_aruco_point2D-carriage)
                    if error<5:
                        l_valid_offsets.append((index,val_a,val_b,val_d,val_h))
