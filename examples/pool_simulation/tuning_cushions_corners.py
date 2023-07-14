import cv2
import time
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1
all_cushions=[]
cushion=[]
counter=1

# Create a function based on a CV2 Event (Left button click)
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,all_cushions,cushion,counter
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # we take note of where that mouse was located
        ix,iy = x,y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        drawing == True
        
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img,(ix,iy),9,(0,0,255), thickness=-1)
        drawing = False
        cushion.append((ix,iy))
        if counter % 4 == 0:
            all_cushions.append(cushion)
            cushion=[]
        counter += 1

img = cv2.imread('./results/warp_corners.jpg')
h,w = img.shape[:2]

# This names the window so we can reference it
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', w//3, h//3)

# Connects the mouse button to our callback function
cv2.setMouseCallback('image',draw_circle)

while(1):
    time.sleep(0.1)
    print(ix,iy)
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

#enhance all_cushions
# cushions are defined clockwise: top left - top right - right - bottom right - bottom left - left 
# corners of cushion are computed clockwise starting at one of the trapezium base vertexs

print(all_cushions)
all_cushions_arr = np.array(all_cushions) # shape should be 6 cushions x 4 corners x 2 dimensions (x and y)
all_cushions_x = all_cushions_arr[...,0]
all_cushions_y = all_cushions_arr[...,1]

cushions_1_and_2_y = all_cushions_y[0:2,:]
cushions_1_and_2_y[:,-2:] = np.mean(cushions_1_and_2_y[:,-2:]) # corners 3 and 4
cushions_1_and_2_y[:,:2] = 0

cushions_3_x = all_cushions_x[2,:]
cushions_3_x[:2] = w
cushions_3_x[-2:] = np.mean(cushions_3_x[-2:])

cushions_4_and_5_y = all_cushions_y[3:5,:]
cushions_4_and_5_y[:,-2:] = np.mean(cushions_4_and_5_y[:,-2:]) # corners 3 and 4
cushions_4_and_5_y[:,:2] = h

cushions_6_x = all_cushions_x[5,:]
cushions_6_x[:2] = 0
cushions_6_x[-2:] = np.mean(cushions_6_x[-2:])

print(all_cushions_arr.tolist())

