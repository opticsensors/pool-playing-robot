import cv2
import numpy as np
from pool.eye import Eye

#read image with random ball config and background
img = cv2.imread('./data/config_3.jpg')
bg = cv2.imread('./data/background.jpg')

eye=Eye()
eye.color_to_lab={
    'white':  [227, 127, 163],
    'yellow': [216, 132, 189],
    'blue':   [89, 134, 103],
    'red':    [147, 181, 181],
    'purple': [78, 137, 119],
    'orange': [180, 165 , 175],
    'green':  [110, 106, 132],
    'burgundy':[121, 154, 154],
    'black':  [58, 128, 130]
}

#eye.lower_lab=[175, 0, 0]
#eye.upper_lab=[255, 147, 164]
eye.lower_hsv=[0, 0, 120]
eye.upper_hsv=[56, 147, 255]

#hardcoded corners: top-right, top-left, bottom-left, bottom-right
corners=eye.get_pool_corners(img)
warp=eye.perspective_transform(img,corners)
warp_bg=eye.perspective_transform(bg,corners)
cv2.imwrite('./results/warp.png', warp)
print('warp done!')

#warp=eye.crop_image(warp, 150,150)
#cv2.imwrite('./results/cropped_warp.png', warp)
#print('cropped_warp done!')

#get a more accurate hsv color of the pool table cloth using only the pool table pixels (wrap img)
hsv=cv2.cvtColor(warp.copy(), cv2.COLOR_BGR2HSV)
bg_hsv=cv2.cvtColor(warp_bg.copy(), cv2.COLOR_BGR2HSV)

lower_color, upper_color = eye.get_cloth_color(hsv,search_width=30)
lower_color=np.array([42,73,5])
upper_color=np.array([99,182,54])

warped_mask,warped_processed =eye.color_segmentation(hsv, lower_color, upper_color)
bg_mask,bg_processed =eye.color_segmentation(bg_hsv, lower_color, upper_color)

cv2.imwrite('./results/warped_mask.png', warped_mask)
cv2.imwrite('./results/warped_processed.png', warped_processed)
cv2.imwrite('./results/bg_processed.png', bg_processed)
print('color_segmentation done!')

warped_without_bg=eye.substract_background(warped_processed,bg_processed)
cv2.imwrite('./results/warped_without_bg.png', warped_without_bg)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
warped_without_bg = cv2.morphologyEx(warped_without_bg, cv2.MORPH_OPEN, kernel,iterations = 2)
cv2.imwrite('./results/warped_without_bg_processed.png', warped_without_bg)
print('background removal done!')

blobs = eye.remove_small_dots(warped_without_bg,connectivity=8) 
cv2.imwrite('./results/blobs.png', blobs)
print('dots removal done!')

numbered_blobs,d_centroids = eye.find_ball_blobs(blobs)
print('find balls done!')

found_balls=cv2.merge((blobs,blobs,blobs))
for label in np.unique(numbered_blobs):
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(blobs.shape, dtype="uint8")
    mask[numbered_blobs == label] = 255

    # Find contours and determine contour area
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(found_balls, [c], -1, (36,255,12), 4)

cv2.imwrite('./results/split_blobs.png', found_balls)

#tune white color
eye.tune_white_color(warp, numbered_blobs)

# tune ball color
eye.tune_ball_color(warp,numbered_blobs,color_space='hsv')

labeled_balls=warp.copy()
if cv2.countNonZero(blobs)!=0:
    sorted_centroids=eye.classify_balls(warp,numbered_blobs,d_centroids,color_space='hsv')

    for ball_num in sorted_centroids:
        x,y=sorted_centroids[ball_num]
        labeled_balls=cv2.putText(labeled_balls.copy(), "#{}".format(ball_num), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
    
    cv2.imwrite('./results/labeled_balls.png', labeled_balls)

print('classify balls done!')
