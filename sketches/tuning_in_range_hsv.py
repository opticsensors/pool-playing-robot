import numpy as np
import cv2
import time

img_num=1

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar("L - H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    time.sleep(0.33)

    img = cv2.imread(f'./masked_{img_num}.png')
    hsv=cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)
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