import numpy as np
import pandas as pd
from pool.eye import Eye
from pool.calibration import CameraCalibration
from sklearn.metrics import r2_score, mean_squared_error

eye=Eye()
calib=CameraCalibration()
rvec,tvec=calib.load_world_frame()
df = pd.read_csv('./results/calibration_dof_to_step.csv', sep=',',decimal='.')
l_carriage_pred=[]

a= 8.906e-01 
b= 3.539e+00  
d=1.160e+00  
h=8.054e+00   

for index, row in df.iterrows():

    target=row[['target_x_undist','target_y_undist']].values
    angle=row[['angle']].values[0]-90
    carriage_pred = calib.predict_carriage_position_in_image_plane(target, angle, (a,b,d,h) ,rvec, tvec)
    l_carriage_pred.append(carriage_pred)

y_test=df[['carriage_x_undist','carriage_y_undist']].values
y_pred=np.array(l_carriage_pred)
print(mean_squared_error(y_test,y_pred,squared=False),r2_score(y_test,y_pred))
