import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pool.calibration import CameraCalibration

df = pd.read_csv('./results/calibration_image_data.csv', sep=',',decimal='.')

calib=CameraCalibration()

x = df[['target_x_undist', 'target_y_undist','angle_undist']]
y = df[['carriage_x_undist','carriage_y_undist']]

def model(params, X):
    # here you need to implement your real model
    # for Predicted_Installation
    a,b,d,h= params
    l_carriage_pred=[]
    for row in X:
        target=row[:2]
        angle=row[2]
        carriage_pred = calib.predict_carriage_position_in_image_plane(target, angle, (a,b,d,h))
        l_carriage_pred.append(carriage_pred)
    y_pred=np.array(l_carriage_pred)
    return y_pred

def sum_of_squares(params, X, Y):
    y_pred = model(params, X)
    obj = np.sqrt(((y_pred - Y) ** 2).sum())
    return obj

# generate some test data
X = x.values
Y = y.values

res = minimize(sum_of_squares, [0.910591,3.442596,1.118040,9.291113], args=(X, Y), tol=1e-3, method="Powell")
print(res)
