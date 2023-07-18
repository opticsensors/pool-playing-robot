import cv2
from pool.ball_detection import Yolo

#read image with random ball config and background
img = cv2.imread('./results/warp_5.png')
yolo=Yolo()
d_centroids, _ = yolo.detect_balls(img,conf=0.25, overlap_threshold=100)
img = yolo.debug(img,d_centroids)
cv2.imwrite('./results/YOLO_Detection.png', img)