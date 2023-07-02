import numpy as np
import cv2
from pool.eye import Eye
from pool.calibration import CameraCalibration,DataExtractor

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

# predict uv carriage given uv ball centroid and angle
img_arucos = cv2.imread('./results/arucos_to_test_3.jpg')
img_arucos = eye.undistort_image(img_arucos, remapping=False)
img_pose=cv2.imread('./results/arucos_to_test_4.jpg')
img_pose = eye.undistort_image(img_pose, remapping=False)
img_corners=cv2.imread('./results/corners_0.jpg')
img_corners = eye.undistort_image(img_corners, remapping=False)
de = DataExtractor(img_corners)
flipper_arucos=[76, 77, 78, 79, 80, 81, 82, 83]
carriage_aruco=22
ball_centroid_arucos=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]
ball_centroids=eye.get_aruco_coordinates_given_several_aruco_ids(img_arucos,ball_centroid_arucos)
carriage_centroid=eye.get_aruco_coordinates_given_aruco_id(img_pose, carriage_aruco)
angle=de.get_flipper_angle(img_pose,flipper_arucos)
uvCarriage=calib.predict_carriage_position_in_image_plane(ball_centroids[14], angle, (0.910591,3.442596,1.118040,9.291113), rvec, tvec)
print(f'real: {carriage_centroid}, pred: {uvCarriage}')

