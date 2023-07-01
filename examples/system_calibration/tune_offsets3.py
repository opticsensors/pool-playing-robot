import numpy as np
import cv2
import pandas as pd
from pool.eye import Eye
from pool.calibration import DataExtractor, CameraCalibration
from sklearn.metrics import r2_score, mean_squared_error

eye=Eye()

img_corners = cv2.imread('./results/corners_0.jpg')
img_corners = eye.undistort_image(img_corners, remapping=False)

calib=CameraCalibration()
de = DataExtractor(img_corners)
rvec,tvec=calib.load_world_frame()
df = pd.read_csv('./results/calibration_image_data.csv', sep=',',decimal='.')
l_errors=[]
l_carriage_pred=[]

a=26.4/25
b=74.1/25
d=22/25
h=220/25 - 19/25

for index, row in df.iterrows():

    target=row[['target_x_undist','target_y_undist']].values
    angle=row[['angle_undist']].values[0]
    carriage_pred = calib.predict_carriage_position_in_image_plane(target, angle, (a,b,d,h) ,rvec, tvec)
    l_carriage_pred.append(carriage_pred)

y_test=df[['carriage_x_undist','carriage_y_undist']].values
y_pred=np.array(l_carriage_pred)
print(mean_squared_error(y_test,y_pred,squared=False),r2_score(y_test,y_pred))
