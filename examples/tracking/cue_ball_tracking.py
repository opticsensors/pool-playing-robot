import time
import cv2
import numpy as np
import pandas as pd
from pool.eye import Eye

eye=Eye()
bg_subtractor=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
frames_to_skip=0
learning_rate=0.01
kernel_size=(25,25)
starting_frame=60 # frame to start tracking (it should be when carriage and flipper already moved)

#frames to compute angle
start=80
end=115

bg_counter=0
frame_counter=0
dict_to_save = {}
list_of_dict = []

cap=cv2.VideoCapture('results/20230710_192549.mp4')
for _ in range(frames_to_skip+1):
    ret, frame0 = cap.read()

corners0 = eye.get_pool_corners(frame0)
matrix0 = eye.calculate_perspective_matrix(corners0)

while(ret):
    ret, frame = cap.read()
    if ret==True and frame_counter>starting_frame:
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
                    cv2.circle(transformed,(cx,cy),8,(255,0,255),-1)     
                    dict_to_save['frame_number']=frame_counter
                    dict_to_save['centroid_x']=cx
                    dict_to_save['centroid_y']=cy
                    list_of_dict.append(dict_to_save.copy())

            fg_mask_3ch=cv2.merge((fg_mask,fg_mask,fg_mask))
            to_show = np.hstack([transformed,fg_mask_3ch])
            cv2.putText(to_show, str(frame_counter),(to_show.shape[1]-200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
            to_show = cv2.resize(to_show, (0,0), fx=0.6, fy=0.6)
            #time.sleep(0.5)
            cv2.imshow('transformed', to_show)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break
    frame_counter+=1

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
line_points=df[['centroid_x', 'centroid_y']][(df['frame_number']>start) & (df['frame_number']<end)].values
[vx,vy,x,y] = cv2.fitLine(line_points,cv2.DIST_L2,0,0.01,0.01)

angle1=np.degrees(np.arctan2(vy,vx))
if angle1<0:
    angle2=180+angle1
else:
    angle2=angle1-180
print(angle1, angle2)

