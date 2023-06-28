import cv2
import numpy as np
from pool.ball_detection import ClassicCV

#read image with random ball config and background
warp = cv2.imread('./results/warp.jpg')
warp_bg = cv2.imread('./results/warp_bg.jpg')
classic_cv=ClassicCV()

#get a more accurate hsv color of the pool table cloth using only the pool table pixels (wrap img)
hsv=cv2.cvtColor(warp.copy(), cv2.COLOR_BGR2HSV)
bg_hsv=cv2.cvtColor(warp_bg.copy(), cv2.COLOR_BGR2HSV)
lower_color, upper_color = classic_cv.get_cloth_color(hsv,search_width=30)
print('auto: ', lower_color, upper_color)

#hardcoded hsv range values are more accurate (obtained with tuning_in_range_cloth_hsv.py)
lower_color=np.array([42,45,2])
upper_color=np.array([100,175,87])
print('manu: ', lower_color, upper_color)

mask,mask_processed =classic_cv.color_segmentation(hsv, lower_color, upper_color)
bg_mask,bg_mask_processed =classic_cv.color_segmentation(bg_hsv, lower_color, upper_color)

cv2.imwrite('./results/mask.png', mask)
cv2.imwrite('./results/mask_processed.png', mask_processed)
cv2.imwrite('./results/bg_mask.png', bg_mask_processed)
cv2.imwrite('./results/bg_mask_processed.png', bg_mask_processed)

mask_without_bg=classic_cv.substract_background(mask_processed,bg_mask_processed)
cv2.imwrite('./results/mask_without_bg.png', mask_without_bg)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
mask_without_bg_processed = cv2.morphologyEx(mask_without_bg, cv2.MORPH_OPEN, kernel,iterations = 2)
cv2.imwrite('./results/mask_without_bg_processed.png', mask_without_bg_processed)

blobs = classic_cv.remove_small_dots(mask_without_bg_processed,connectivity=8) 
cv2.imwrite('./results/blobs.png', blobs)

numbered_blobs,d_centroids = classic_cv.find_ball_blobs(blobs)
blobs_with_contours_drawn = classic_cv.debug_find_ball_blobs(blobs, numbered_blobs)
cv2.imwrite('./results/blobs_with_contours_drawn.png', blobs_with_contours_drawn)


#tune white color
classic_cv.tune_white_color(warp, numbered_blobs)


# tune ball color
classic_cv.tune_ball_color(warp,numbered_blobs,color_space='hsv')


labeled_balls=warp.copy()
if cv2.countNonZero(blobs)!=0:
    sorted_centroids=classic_cv.classify_balls(warp,numbered_blobs,d_centroids,color_space='hsv')

    for ball_num in sorted_centroids:
        x,y=sorted_centroids[ball_num]
        labeled_balls=cv2.putText(labeled_balls.copy(), "#{}".format(ball_num), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
    
    cv2.imwrite('./results/CLASSICCV_Detection.png', labeled_balls)

