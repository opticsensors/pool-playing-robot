import cv2
import numpy as np
from pool.eye import Eye
from pool.ball_detection import Yolo

eye=Eye()
yolo=Yolo()

# Prespective matrix
img_corners=cv2.imread('./results/img_0.jpg')
img_corners_undist=eye.undistort_image(img_corners, remapping=False)
undist_corners=eye.get_pool_corners(img_corners_undist)
undist_matrix=eye.calculate_perspective_matrix(undist_corners)

# undist warp img
img=cv2.imread('./results/img_0.jpg')
img_undist=eye.undistort_image(img, remapping=False)
img_undist_warp=eye.transform_image_given_a_matrix(img_undist, undist_corners, undist_matrix)

# find centroids using both methods
d_centroids_undist, _ = yolo.detect_balls(img_undist,conf=0.25, overlap_threshold=100)
d_centroids_undist_warp, _ = yolo.detect_balls(img_undist_warp,conf=0.25, overlap_threshold=100)

d_centroids_undist_warp2 = {}
for ball_num,point in d_centroids_undist.items():
    d_centroids_undist_warp2[ball_num] = eye.transform_point_given_a_matrix(point, undist_matrix)

# compute differences
d_errors = {x: np.linalg.norm(np.array(d_centroids_undist_warp[x]) - np.array(d_centroids_undist_warp2[x])) for x in d_centroids_undist_warp if x in d_centroids_undist_warp2}

def debug(img, d_centroids, c):
    for ball_num in d_centroids:
        x,y=d_centroids[ball_num]
        img=cv2.circle(img, (int(x), int(y)), 8, c, -1)
    return img
img1 = debug(img_undist_warp,d_centroids_undist_warp, (0,255,0))
img2 = debug(img1,d_centroids_undist_warp2,(0,0,255))
cv2.imwrite('./results/balls_undist_warp.jpg', img2)