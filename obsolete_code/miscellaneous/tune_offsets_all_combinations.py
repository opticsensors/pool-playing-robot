import numpy as np
import pandas as pd
from pool.eye import Eye
from pool.calibration import CameraCalibration
from sklearn.metrics import r2_score, mean_squared_error

eye=Eye()
calib=CameraCalibration()
rvec,tvec=calib.load_world_frame()
df = pd.read_csv('./results/calibration_image_data.csv', sep=',',decimal='.')
l_carriage_pred=[]
log=[]

a=0.910591
b=3.442596
d=1.118040
h=9.291113

width_a=0.02*a
width_b=0.02*b
width_h=0.02*h
width_d=0.02*d
range_a=np.linspace(a-width_a, a+width_a, num=7)
range_b=np.linspace(b-width_b, b+width_b, num=7)
range_h=np.linspace(h-width_h, h+width_h, num=1)
range_d=np.linspace(d-width_d, d+width_d, num=7)
c=0

for val_a in range_a:
    c=c+1
    print('======================================', ' iter:', c, ' ======================================')
    for val_b in range_b:
        for val_d in range_d:
            for val_h in range_h:

                for index, row in df.iterrows():

                    target=row[['target_x_undist','target_y_undist']].values
                    angle=row[['angle_undist']].values[0]
                    carriage_pred = calib.predict_carriage_position_in_image_plane(target, angle, (val_a,val_b,val_d,val_h) ,rvec, tvec)
                    l_carriage_pred.append(carriage_pred)

                y_test=df[['carriage_x_undist','carriage_y_undist']].values
                y_pred=np.array(l_carriage_pred)
                log.append((val_a,val_b,val_d,val_h,mean_squared_error(y_test,y_pred,squared=False),r2_score(y_test,y_pred)))
                l_carriage_pred=[]

log=np.array(log)
df=pd.DataFrame({'a':log[:,0],
                'b':log[:,1],
                'd':log[:,2],
                'h':log[:,3],
                'rmse':log[:,4],
                'r_2':log[:,5]})

df.to_csv(path_or_buf='./results/calibration_offsets.csv', sep=',',index=False)