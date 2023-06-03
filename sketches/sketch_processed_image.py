import cv2
from pool.eye import Eye
import numpy as np

#CV algorithms initialization
eye=Eye()

# We undistort the image
img            = cv2.imread(f'./data/corners_0.jpg')
cameraMatrix   = np.load('./data/cameraMatrix.npy')
dist           = np.load('./data/dist.npy')
undist_img     = eye.undistort_image(
    img,
    cameraMatrix,
    dist,
    remapping=False,
)
undist_corners = eye.get_pool_corners(
    undist_img,
    bottom_aruco_ids = [6,7],
    top_aruco_ids    = [2,3],
    left_aruco_ids   = [0,1],
    right_aruco_ids  = [4,5],
)
prespective_matrix = eye.calculate_perspective_matrix(undist_corners)
warp_undistorted   = eye.transform_image_given_a_matrix(
    undist_img,
    undist_corners,
    prespective_matrix,
)

cv2.imwrite('./results/warp_corners.jpg', warp_undistorted)