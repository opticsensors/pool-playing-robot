import cv2
import numpy as np
import pandas as pd

data = pd.read_csv('./data/calibration_image_data.csv', sep=' ',decimal='.')
img = cv2.imread(f'./stepper_calibration/img_0.jpg')
points=np.hstack([data['x_dist'].values.reshape(-1,1), data['y_dist'].values.reshape(-1,1)])  
points=points.astype(np.int32)
for point in points:
    cv2.circle(img,tuple(point),10,(0,0,255),-1)

cv2.imwrite('./results/pixel_to_step.png', img)