import cv2
from pool.ball_detection import ClassicCV

classic_cv=ClassicCV()

#read image with random ball config and background
img = cv2.imread('./results/warp_5.png')
bg = cv2.imread('./results/warp_bg.png')

d_centroids= classic_cv.detect_balls(img, bg, cloth_color='tuned', color_space='hsv', debug_path='./results/', debug=True)
img = classic_cv.debug(img,d_centroids)
cv2.imwrite('./results/CLASSICCV_Detection.png', img)
