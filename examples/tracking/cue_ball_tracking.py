import cv2
import numpy as np
from pool.eye import Eye

eye=Eye()
bg_subtractor=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
frames_to_skip=0
learning_rate=0.0
kernel_size=(25,25)

cap=cv2.VideoCapture('results/20230710_192549.mp4')
for _ in range(frames_to_skip+1):
    ret0, frame0 = cap.read()

corners0 = eye.get_pool_corners(frame0)
matrix0 = eye.calculate_perspective_matrix(corners0)

while(True):
    ret, frame = cap.read()
    transf = eye.transform_image_given_a_matrix(frame, corners0, matrix0)
    result = cv2.cvtColor(transf, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(result, kernel_size, 0)
    result = bg_subtractor.apply(result, learningRate=learning_rate)

    contours, hier = cv2.findContours(result, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(transf,(cx,cy),8,(255,0,255),-1)

    transf = cv2.resize(transf, (0,0), fx=0.6, fy=0.6)
    cv2.imshow('transf', transf)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
