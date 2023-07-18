import cv2
from pool.eye import Eye

#CV algorithms initialization
eye=Eye()

# We undistort the image
img            = cv2.imread(f'./results/img_1.jpg')
undist_img     = eye.undistort_image(img,remapping=False)
corners = eye.get_pool_corners(img)
undist_corners = eye.get_pool_corners(undist_img)
prespective_matrix = eye.calculate_perspective_matrix(corners)
undist_prespective_matrix = eye.calculate_perspective_matrix(undist_corners)
warp_distorted   = eye.transform_image_given_a_matrix(img,corners,prespective_matrix)
warp_undistorted   = eye.transform_image_given_a_matrix(undist_img,undist_corners,undist_prespective_matrix)

cv2.imwrite('./results/img_1_undist.jpg', undist_img)
cv2.imwrite('./results/img_1_dist_warp.jpg', warp_distorted)
cv2.imwrite('./results/img_1_undist_warp.jpg', warp_undistorted)