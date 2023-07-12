import cv2
import numpy as np
from pool.eye import Eye

eye=Eye()
bg_subtractor=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
frames_to_skip=0
learning_rate=0.01
kernel_size=(25,25)
bg_counter=0
name='20230711_141720_1'

cap=cv2.VideoCapture(f'results/{name}.mp4')
for _ in range(frames_to_skip+1):
    ret, frame0 = cap.read()

corners0 = eye.get_pool_corners(frame0)
#corners0=[(200,1403), (194,163), (886,163), (875,1404)] # only for video 20230711_174744_1
#corners0=[(202,1407), (178,175), (868,165), (874,1403)] # only for video 20230711_141105_1
#corners0=[(198,1408), (196,173), (887,175), (871,1414)] # only for video 20230711_141720_1

matrix0 = eye.calculate_perspective_matrix(corners0)
transformed_frame0 = eye.transform_image_given_a_matrix(frame0, corners0, matrix0)

while(ret):
    ret, frame = cap.read()
    if ret==True:
        transformed = eye.transform_image_given_a_matrix(frame, corners0, matrix0)
        transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        transformed_blur = cv2.GaussianBlur(transformed_gray, kernel_size, 0)
        fg_mask = bg_subtractor.apply(transformed_blur, learningRate=learning_rate)

        if bg_counter < 25:
            bg_counter += 1
            pass
        else:
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
            contours, hier = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                bigger_contour = max(contours, key = cv2.contourArea)
                max_area = cv2.contourArea(bigger_contour)
                if max_area>2000:
                    M = cv2.moments(bigger_contour)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(transformed_frame0,(cx,cy),5,(255,0,255),-1)     
                    cv2.circle(transformed,(cx,cy),8,(255,0,255),-1)     

            fg_mask_3ch=cv2.merge((fg_mask,fg_mask,fg_mask))
            to_show = np.hstack([transformed,fg_mask_3ch])
            to_show = cv2.resize(to_show, (0,0), fx=0.6, fy=0.6)
            cv2.imshow('transformed', to_show)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break

cv2.imwrite(f'./results/{name}.png', transformed_frame0)
cap.release()
cv2.destroyAllWindows()


