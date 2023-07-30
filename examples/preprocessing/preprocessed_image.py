import cv2
from pool.eye import Eye

#CV algorithms initialization
eye=Eye()

# We undistort the image
img_name = 'img_0'
img            = cv2.imread(f'./results/{img_name}.jpg')
undist_img     = eye.undistort_image(img,remapping=False)
corners = eye.get_pool_corners(img)
undist_corners = eye.get_pool_corners(undist_img)
prespective_matrix = eye.calculate_perspective_matrix(corners)
undist_prespective_matrix = eye.calculate_perspective_matrix(undist_corners)
warp_distorted   = eye.transform_image_given_a_matrix(img,corners,prespective_matrix)
warp_undistorted   = eye.transform_image_given_a_matrix(undist_img,undist_corners,undist_prespective_matrix)

cv2.imwrite(f'./results/{img_name}_undist.jpg', undist_img)
cv2.imwrite(f'./results/{img_name}_dist_warp.jpg', warp_distorted)
cv2.imwrite(f'./results/{img_name}_undist_warp.jpg', warp_undistorted)