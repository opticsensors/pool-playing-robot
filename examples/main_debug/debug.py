import cv2
from pool.calibration import InverseKinematics
from pool.ball_detection import Yolo
from pool.brain import ShotSelection
from pool.eye import Eye

print("Done importing!")

#necessary objects
ik = InverseKinematics()
yolo = Yolo()
ss = ShotSelection()
eye = Eye()

turn='solid'

img = cv2.imread('./results/img_0.jpg')
img_undist_warp = eye.undistort_and_warp_image(img)
cv2.imwrite('./results/img_undist_warp_0.png', img_undist_warp)

d_centroids, _ = yolo.detect_balls(img_undist_warp,conf=0.25, overlap_threshold=100)
img_yolo_debug = yolo.debug(img_undist_warp, d_centroids)
cv2.imwrite('./results/img_yolo_debug.png', img_yolo_debug)

uvBallCentroid = d_centroids[0]
angle = ss.get_actuator_angle(d_centroids, turn)
img_brain_debug,df = ss.debug(img_undist_warp, d_centroids, turn, shot_type='CTP')
cv2.imwrite('./results/img_brain_debug.png', img_brain_debug)

angle_dxl = angle + 90 
steps1,steps2=ik.img_data_to_steps(uvBallCentroid, angle_dxl)
steps1=int(steps1)
steps2=int(steps2)
