import cv2
from pool.eye import Eye

#read image with random ball config
img = cv2.imread('./data/random_config.jpg')
eye=Eye()
eye.color_to_lab={
    'white': [227, 130, 148],
    'yellow': [216.18407452, 124.80355131, 195.73519079],
    'blue': [89.64986175, 134.93342045, 103.84520826],
    'red': [147.45829885, 172.49603901, 178.22393935],
    'purple': [78.6311673, 135.5586135, 123.057679],
    'orange': [180.62554741, 151.8795586 , 187.98600759],
    'green': [110.05696026, 106.50088757, 132.41375192],
    'burgundy': [121.56706638, 160.19773476, 158.41379201],
    'black': [58, 129, 136]
}

eye.lower_lab=[175, 0, 0]
eye.upper_lab=[255, 147, 164]
eye.lower_hsv=[0, 0, 179]
eye.upper_hsv=[180, 106, 255]

#hardcoded corners: top-right, top-left, bottom-left, bottom-right
corners=[(4396,667),(282,651),(134,3056),(4525,3059)]
warp=eye.perspective_transform(img,corners)

#get a more accurate hsv color of the pool table cloth using only the pool table pixels (wrap img)
hsv=cv2.cvtColor(warp.copy(), cv2.COLOR_BGR2HSV)
lower_color, upper_color = eye.get_cloth_color(hsv,search_width=25)
warped_mask,warped_median=eye.color_segmentation(hsv, lower_color, upper_color,filter_radius=17)
cv2.imwrite('./results/warped_median.png', warped_median)

numbered_single_blobs, single_centroids, single_blobs,connected_blobs = eye.find_ball_bolbs(warped_median,connectivity=8,min_size=20000, max_size=40000,thresh_convexity=0.95, thresh_roundness=0.85)
cv2.imwrite('./results/single_blobs.png', single_blobs)
cv2.imwrite('./results/connected_blobs.png', connected_blobs)

labeled_balls=warp.copy()
if cv2.countNonZero(single_blobs)!=0:
    sorted_single_centroids=eye.classify_balls(warp,numbered_single_blobs,single_centroids,color_space='hsv')

    for ball_num in sorted_single_centroids:
        x,y=sorted_single_centroids[ball_num]
        labeled_balls=cv2.putText(labeled_balls.copy(), "#{}".format(ball_num), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imwrite('./results/labeled_balls.png', labeled_balls)

if cv2.countNonZero(connected_blobs)!=0:
    numbered_connected_blobs,connected_centroids=eye.split_connected_balls_v2(connected_blobs)
    sorted_connected_centroids=eye.classify_balls(img,numbered_connected_blobs,connected_centroids,color_space='hsv')

    for ball_num in sorted_connected_centroids:
        x,y=sorted_connected_centroids[ball_num]
        labeled_balls=cv2.putText(labeled_balls.copy(), "#{}".format(ball_num), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite('./results/labeled_balls.png', labeled_balls)