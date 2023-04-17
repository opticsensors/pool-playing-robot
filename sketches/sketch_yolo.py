import cv2
import numpy as np
from pool.eye import Eye

#read image with random ball config and background
img = cv2.imread('./data/config_1.jpg')

eye=Eye()
#corners=eye.get_pool_corners(img,bottom_aruco_ids=[17,8],top_aruco_ids=[9,3],left_aruco_ids=[1,11],right_aruco_ids=[23,10])
#warp=eye.perspective_transform(img,corners)

# Yolov8 results
d_centroids, _ =eye.YOLO(img,conf=0.25, overlap_threshold=100, data_path='./data/data.yaml',model_path='./data/yolov8m.pt')
for ball_num in d_centroids:
    x,y=d_centroids[ball_num]
    img=cv2.putText(img, "#{}".format(ball_num), (int(x) - 10, int(y)),
    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    img=cv2.circle(img, (int(x), int(y)), 8, (255, 0, 255), -1)

cv2.imwrite('./results/chosen_YOLO_Detection.jpg', img)