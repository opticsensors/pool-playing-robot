import numpy as np
import cv2
import time

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar("Lower - H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Lower - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Lower - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Upper - H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("Upper - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper - V", "Trackbars", 255, 255, nothing)

while True:
    time.sleep(0.33)

    img = cv2.imread(f'./results/results_tune_white.png')
    hsv=cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("Lower - H", "Trackbars")
    l_s = cv2.getTrackbarPos("Lower - S", "Trackbars")
    l_v = cv2.getTrackbarPos("Lower - V", "Trackbars")
    u_h = cv2.getTrackbarPos("Upper - H", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper - S", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper - V", "Trackbars")
    
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)
    mask_3ch=cv2.merge((mask,mask,mask))

    result = np.hstack((img, mask_3ch))
    result = cv2.resize(result, (0,0), fx=0.66, fy=0.66)
    cv2.imshow("mask", result)

    key=cv2.waitKey(1)
    if key==27:
        break

cv2.destroyAllWindows()