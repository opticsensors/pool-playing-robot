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

# predict uv carriage given uv ball centroid and angle
img_corners=cv2.imread('./results/corners_0.jpg')
img_corners = eye.undistort_image(img_corners, remapping=False)
de=DataExtractor(img_corners)

uv1=np.array([2500 ,  2000, 1])
uv2=np.array([294 ,  1600, 1])
vec1=uv2[:2]-uv1[:2]

uv1_warp=eye.transform_point_given_a_matrix(uv1[:2], de.undist_matrix)
uv2_warp=eye.transform_point_given_a_matrix(uv2[:2], de.undist_matrix)
vec2=np.array(uv2_warp)-np.array(uv1_warp)

xyz1=calib.transform_image_point_to_world_point(uv1.reshape(-1,1), 0 ,rvec, tvec)
xyz2=calib.transform_image_point_to_world_point(uv2.reshape(-1,1), 0 ,rvec, tvec)
vec3=xyz2[:2]-xyz1[:2]

angle_undist=np.degrees(np.arctan2(vec1[1],vec1[0]))
angle_undist_warp=np.degrees(np.arctan2(vec2[1],vec2[0]))
angle_real_world=np.degrees(np.arctan2(vec3[1],vec3[0]))+90

print(angle_undist,angle_undist_warp,angle_real_world)