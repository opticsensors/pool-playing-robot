import cv2
from pool.ball_detection import Yolo
import time

#read image with random ball config and background
img = cv2.imread('./results/warp_9.png')
yolo=Yolo()

start_time = time.time()
d_centroids, _ = yolo.detect_balls(img,conf=0.25, overlap_threshold=100)
print("--- %s seconds ---" % (time.time() - start_time))

img = yolo.debug(img,d_centroids)
cv2.imwrite('./results/YOLO_Detection.png', img)