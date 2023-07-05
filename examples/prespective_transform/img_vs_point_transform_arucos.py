import cv2
import numpy as np
from pool.eye import Eye

eye=Eye()

# Prespective matrix
img_corners=cv2.imread('./results/img_0.jpg')
img_corners_undist=eye.undistort_image(img_corners, remapping=False)
undist_corners=eye.get_pool_corners(img_corners_undist)
undist_matrix=eye.calculate_perspective_matrix(undist_corners)

# undist warp img
img=cv2.imread('./results/img_arucos.jpg')
img_undist=eye.undistort_image(img, remapping=False)
img_undist_warp=eye.transform_image_given_a_matrix(img_undist, undist_corners, undist_matrix)

# find centroids using both methods
d_centroids_undist = eye.get_aruco_coordinates_given_several_aruco_ids(img_undist,arucos_to_track=[13,15,17,18,20,21])

d_centroids_undist_warp = {}
for ball_num,point in d_centroids_undist.items():
    d_centroids_undist_warp[ball_num] = eye.transform_point_given_a_matrix(point, undist_matrix)

def debug(img, d_centroids, c):
    for ball_num in d_centroids:
        x,y=d_centroids[ball_num]
        img=cv2.circle(img, (int(x), int(y)), 8, c, -1)
    return img
img1 = debug(img_undist,d_centroids_undist, (0,255,0))
img2 = debug(img_undist_warp,d_centroids_undist_warp, (0,255,0))
cv2.imwrite('./results/arucos_undist.jpg', img1)
cv2.imwrite('./results/arucos_undist_warp.jpg', img2)