import cv2
import numpy as np
from pool.ball_detection import ClassicCV

cv=ClassicCV()

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

a=218
b=203
img=np.zeros((3*a,3*a), dtype=np.uint8)
h, w = img.shape

cv2.ellipse(img,(w//2,h//2),(a,b),0,0,360,255,-1)
ellipses=[]
for angle in range(0,360,10):
    ellipses.append(rotate_image(img, angle))

ellipses_to_draw=[]
titles=[]
for ellipse in ellipses:
    contours,_ = cv2.findContours(ellipse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c=contours[0]
    rect = cv2.boundingRect(c)
    rect_corners=(int(rect[0]), int(rect[1])),(int(rect[0]+rect[2]), int(rect[1]+rect[3]))
    rect_centroid=(int(rect[0])+int(rect[2]/2), int(rect[1])+int(rect[3]/2))
    M = cv2.moments(c)
    blob_centroid = (M["m10"] / M["m00"]),(M["m01"] / M["m00"])
    ellipse_3ch=cv2.merge((ellipse,ellipse,ellipse))
    cv2.rectangle(ellipse_3ch, rect_corners[0], rect_corners[1], (0,255,0), 2)
    cv2.circle(ellipse_3ch, (rect_centroid[0], rect_centroid[1]), 4, (0, 0, 255), -1)
    cv2.circle(ellipse_3ch, (int(blob_centroid[0]), int(blob_centroid[1])), 4, (255, 0, 0), -1)
    error = np.linalg.norm(np.array(blob_centroid) - np.array(rect_centroid))
    ellipses_to_draw.append(ellipse_3ch)
    titles.append("{:.3f}".format(error))


grid=cv.display_images_of_same_size_in_a_grid(ellipses_to_draw, grid_size=(6,6), titles=titles)
cv2.imwrite('./results/ellipse.png', grid)
