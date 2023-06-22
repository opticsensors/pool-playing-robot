import cv2
from pool.eye import Eye
import numpy as np

img = cv2.imread('./results/end_effector.jpg')
eye=Eye()
x,y=eye.get_aruco_coordinates(img, 22)
cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)

cv2.imwrite('./results/arucos_to_detect.png', img)
