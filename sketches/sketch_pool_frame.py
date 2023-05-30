import numpy as np
import cv2

img = cv2.imread(f'./results/warp_0.jpg')
H=img.shape[0]
W=img.shape[1]

def draw_cross(img,point,length):
    x,y=point
    cv2.line(img, (x+length, y), (x-length, y), (0, 255, 255), thickness=8)
    cv2.line(img, (x, y-length), (x, y+length), (0, 255, 255), thickness=8)
    return img

x_pocket_corners=120
y_pocket_corners=96
radius_pocket_corners=210

x_pocket_middle=2430
y_pocket_middle=164
radius_pocket_middle=112

pocket1=(x_pocket_corners,y_pocket_corners)
pocket2=(W-x_pocket_corners,y_pocket_corners)
pocket3=(W-x_pocket_corners,H-y_pocket_corners)
pocket4=(x_pocket_corners,H-y_pocket_corners)
pocket5=(x_pocket_middle,y_pocket_middle)
#pocket6=(x_pocket_middle,H-y_pocket_middle)
pocket6=(2427,2574)

horizontal_line_distance=250
vertical_line_distance=260
epsilon_corner=80
epsilon_middle=70

cv2.circle(img, pocket1, radius_pocket_corners, (0, 255, 255), 8)
cv2.circle(img, pocket2, radius_pocket_corners, (0, 255, 255), 8)
cv2.circle(img, pocket3, radius_pocket_corners, (0, 255, 255), 8)
cv2.circle(img, pocket4, radius_pocket_corners, (0, 255, 255), 8)
cv2.circle(img, pocket5, radius_pocket_middle, (0, 255, 255), 8)
cv2.circle(img, pocket6, radius_pocket_middle, (0, 255, 255), 8)

draw_cross(img, pocket1, radius_pocket_corners)
draw_cross(img, pocket2, radius_pocket_corners)
draw_cross(img, pocket3, radius_pocket_corners)
draw_cross(img, pocket4, radius_pocket_corners)
draw_cross(img, pocket5, radius_pocket_middle)
draw_cross(img, pocket6, radius_pocket_middle)

cv2.line(img, (horizontal_line_distance+epsilon_corner, vertical_line_distance), (W-horizontal_line_distance-epsilon_corner, vertical_line_distance), (0, 255, 0), thickness=3)
cv2.line(img, (W-horizontal_line_distance, vertical_line_distance+epsilon_corner), (W-horizontal_line_distance, H-vertical_line_distance-epsilon_corner), (0, 255, 0), thickness=3)
cv2.line(img, (W-horizontal_line_distance-epsilon_corner, H-vertical_line_distance), (horizontal_line_distance+epsilon_corner, H-vertical_line_distance), (0, 255, 0), thickness=3)
cv2.line(img, (horizontal_line_distance, H-vertical_line_distance-epsilon_corner), (horizontal_line_distance, vertical_line_distance+epsilon_corner), (0, 255, 0), thickness=3)

cv2.line(img, (horizontal_line_distance+epsilon_corner, vertical_line_distance), (horizontal_line_distance, vertical_line_distance+epsilon_corner), (255, 255, 0), thickness=20)
cv2.line(img, (W-horizontal_line_distance-epsilon_corner, vertical_line_distance), (W-horizontal_line_distance, vertical_line_distance+epsilon_corner), (255, 255, 0), thickness=20)
cv2.line(img, (horizontal_line_distance, H-vertical_line_distance-epsilon_corner),(horizontal_line_distance+epsilon_corner, H-vertical_line_distance) , (255, 255, 0), thickness=20)
cv2.line(img, (W-horizontal_line_distance-epsilon_corner, H-vertical_line_distance),(W-horizontal_line_distance, H-vertical_line_distance-epsilon_corner) , (255, 255, 0), thickness=20)

cv2.line(img, (x_pocket_middle-epsilon_middle, vertical_line_distance),(x_pocket_middle+epsilon_middle, vertical_line_distance) , (255, 255, 0), thickness=20)
cv2.line(img, (x_pocket_middle-epsilon_middle, H-vertical_line_distance),(x_pocket_middle+epsilon_middle, H-vertical_line_distance) , (255, 255, 0), thickness=20)

cv2.imwrite('./results/pool_frame.png', img)
