import numpy as np
import cv2
import time

img_num=1

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar("L - l", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - a", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - b", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - l", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - a", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - b", "Trackbars", 255, 255, nothing)

while True:
    time.sleep(0.33)

    img = cv2.imread(f'./masked_{img_num}.png')
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    l_l = cv2.getTrackbarPos("L - l", "Trackbars")
    l_a = cv2.getTrackbarPos("L - a", "Trackbars")
    l_b = cv2.getTrackbarPos("L - b", "Trackbars")
    u_l = cv2.getTrackbarPos("U - l", "Trackbars")
    u_a = cv2.getTrackbarPos("U - a", "Trackbars")
    u_b = cv2.getTrackbarPos("U - b", "Trackbars")
    
    lower = np.array([l_l, l_a, l_b])
    upper = np.array([u_l, u_a, u_b])
    mask = cv2.inRange(lab, lower, upper)
    mask_3ch=cv2.merge((mask,mask,mask))

    result = np.hstack((img, mask_3ch))

    cv2.imshow("mask", result)#cv2.resize(thresh, (0,0), fx=0.33, fy=0.33))

    key=cv2.waitKey(1)
    if key==27:
        break

    elif key == 32: # goto next img on space
        img_num+=1
        if img_num>16:
            break

cv2.destroyAllWindows()