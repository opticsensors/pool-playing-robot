import cv2
import numpy as np

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 120)
#cap.set(cv2.CAP_PROP_EXPOSURE, -8)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3,640)
cap.set(4,480)

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar("L - H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while(True):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame,(5,5),0)
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
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

    result = np.hstack((frame, mask_3ch))
    #result = cv2.resize(result, (0,0), fx=0.66, fy=0.66)
    cv2.imshow("mask", result)

    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()