import cv2
from pool.ball_detection import ClassicCV
import time

classic_cv=ClassicCV()

#read image with random ball config and background
img = cv2.imread('./results/warp_9.png')
bg = cv2.imread('./results/warp_bg.png')

start_time = time.time()
d_centroids= classic_cv.detect_balls(img, bg, cloth_color='tuned', color_space='hsv', debug_path='./results/', debug=True)
print("--- %s seconds ---" % (time.time() - start_time))

img = classic_cv.debug(img,d_centroids)
cv2.imwrite('./results/CLASSICCV_Detection.png', img)
