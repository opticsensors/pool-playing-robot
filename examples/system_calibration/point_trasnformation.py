import numpy as np
import cv2
from pool.eye import Eye
from pool.calibration import CameraCalibration

eye=Eye()
calib=CameraCalibration()

img = cv2.imread("./results/checkerboard.jpg")
img = eye.undistort_image(img, remapping=False)
objp,imgp=calib.find_chessboard_corners(img, (14,21))
rvec,tvec=calib.generate_and_save_world_frame(img, (14,21))

# uv to xyz
uvPoint=np.array([3950 ,  912, 1]).reshape(-1,1)
xyzPoint=calib.transform_image_point_to_world_point(uvPoint, 0 ,rvec, tvec)
print(f'uv to xyz: {uvPoint.ravel()}->{xyzPoint}')

# xyz to uv
xyzOrigin=np.array([0., 0., 0.])
uvPoint=calib.transform_world_point_to_img_point(xyzOrigin ,rvec, tvec)
print(f'xyz to uv: {xyzOrigin}->{uvPoint}')
