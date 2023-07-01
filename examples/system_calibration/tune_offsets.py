import numpy as np
import cv2
from pool.eye import Eye
from pool.calibration import DataExtractor, CameraCalibration

eye=Eye()

# Find Carriage aruco point expressend in 3D world frame coordinates:
# img_arucos -> an image where all ball centroid arucos can be detected
# img_pose -> an image where the carriage aruco is in the desired position (flipper on top of a ball centroid and with desired direction)
img_arucos = cv2.imread('./results/arucos_to_test_3.jpg')
img_arucos = eye.undistort_image(img_arucos, remapping=False)
img_pose=cv2.imread('./results/arucos_to_test_4.jpg')
img_pose = eye.undistort_image(img_pose, remapping=False)
img_corners = cv2.imread('./results/corners_0.jpg')
img_corners = eye.undistort_image(img_corners, remapping=False)

calib=CameraCalibration()
de = DataExtractor(img_corners)

flipper_arucos=[76, 77, 78, 79, 80, 81, 82, 83]
carriage_aruco=22
ball_centroid_arucos=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]
d_ball_centroids=eye.get_aruco_coordinates_given_several_aruco_ids(img_arucos,ball_centroid_arucos)
carriage_aruco=eye.get_aruco_coordinates_given_aruco_id(img_pose, carriage_aruco)
rvec,tvec=calib.load_world_frame()

angle=de.get_flipper_angle(img_pose,flipper_arucos)

uvBallCentroid=np.pad(d_ball_centroids[14], (0, 1), constant_values=1).reshape(-1,1)
trans=calib.transform_image_point_to_world_point(uvBallCentroid, 0, rvec, tvec) #translation vector from world frame to auxiliar frame 

angle=angle-90 # conversion angle uv frame to angle auxiliar frame (the frame with x axis with the same direction as flipper)
angle=np.radians(angle)
Rt=np.array([[np.cos(angle), -np.sin(angle), 0, trans[0]],
            [np.sin(angle), np.cos(angle), 0, trans[1]],
            [0, 0, 1, trans[2]],
            [0, 0, 0, 1]])
a=26.4/25
b=74.1/25
h=220/25 - 19/25
d=22/25
offset_point_of_contact=np.array([-b,-a,-h,1]).reshape(-1,1)
offset_rotation_axis=np.array([d,0,0,1]).reshape(-1,1)
carriage_aruco_point3D = (Rt@offset_point_of_contact)[:3,0] + offset_rotation_axis[:3,0]
carriage_aruco_point2D = calib.transform_world_point_to_img_point(carriage_aruco_point3D, rvec, tvec)

width_a=0.5
width_b=1.3
width_h=2
width_d=0.3
range_a=np.linspace(a-width_a, a+width_a, num=30)
range_b=np.linspace(b-width_b, b+width_b, num=30)
range_h=np.linspace(h-width_h, h+width_h, num=30)
range_d=np.linspace(d-width_d, d+width_d, num=30)

for a in range_a:
    for b in range_b:
        for d in range_d:
            for h in range_h:
                offset_point_of_contact=np.array([-b,-a,-h,1]).reshape(-1,1)
                offset_rotation_axis=np.array([d,0,0,1]).reshape(-1,1)
                carriage_aruco_point3D = (Rt@offset_point_of_contact)[:3,0] + offset_rotation_axis[:3,0]
                carriage_aruco_point2D = calib.transform_world_point_to_img_point(carriage_aruco_point3D ,rvec, tvec)
                error=np.linalg.norm(carriage_aruco_point2D-np.array(carriage_aruco))
                if error<10:
                    print(a,b,d,h,carriage_aruco_point2D, error)
