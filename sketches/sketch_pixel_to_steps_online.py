from pool.cam import Camera_DLSR, Camera_DLSR_settings
import time
import cv2
from pool.eye import Eye
import numpy as np
from pool.controller_actuators import Controller_actuators
import pandas as pd
import time

#camera initialization
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
test_camera = Camera_DLSR(control_cmd_location=camera_control_cmd_path)
test_setting: Camera_DLSR_settings = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
test_camera.save_folder='./stepper_calibration/'
test_camera.collection_name = 'img'

#CV algorithms initialization
eye=Eye()

#stepper motor initialization
stp=Controller_actuators(baudRate=9600,serialPortName='COM3')
stp.setupSerial()

#helper functions to make code more readable
def generate_grid(num_horizontal_points, num_vertical_points):
    alpha = 1/(num_horizontal_points+1)
    beta = 1/(num_vertical_points+1)
    points = np.mgrid[1:num_horizontal_points+1,1:num_vertical_points+1].T.reshape(-1,2)
    points = points.astype(np.float32)
    rescaled_points = np.zeros_like(points)
    rescaled_points[:,0] = alpha*points[:,0]
    rescaled_points[:,1] = beta*points[:,1]
    return rescaled_points

def cm_to_steps(incr_x, incr_y, W, H):
    scaler=0.00794
    phi1 = 1/(2*scaler) * (-incr_x*W+incr_y*H)
    phi2 = 1/(2*scaler) * (incr_x*W+incr_y*H)
    return phi1,phi2

#load precalibrated data
cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')

#compute corners ---------------------- picture corners.jpg should be updated in another script or in here in case the structure moves
img=cv2.imread(f'./data/corners.jpg')
undist_img=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
dist_corners=eye.get_pool_corners(img, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])
undist_corners=eye.get_pool_corners(undist_img, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])
#corners_to_transform = np.array(dist_corners, dtype='float32')
#corners_to_transform = np.array([corners_to_transform])
#undist_corners_v2=cv2.undistortPoints(corners_to_transform, cameraMatrix, dist, None, cameraMatrix)
#h,  w = img.shape[:2]
#newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
#undist_corners_v3=cv2.undistortPoints(corners_to_transform, cameraMatrix, dist,None, newCameraMatrix)

#compute prespective transform matrix
# the prespective trans matrix should be the same for all the captured images
_,dist_matrix=eye.perspective_transform(img,dist_corners)
_,undist_matrix=eye.perspective_transform(undist_img,undist_corners)

#define needed variables
points=generate_grid(4,3)
np.random.shuffle(points) 
np.save('./data/calibration_points.npy', points)
mode = 0
count = 0
dict_to_save = {}
list_of_dict = []
pos1 = 0
pos2 = 0
prev_point_x=0
prev_point_y=0
W = 84
H = 44
aruco_to_track=23
stp.sendToArduino(f"-1,0,0")

while True:
    if count>=points.shape[0]:
        break

    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        time.sleep(2)
        # take, read and undistort image
        test_camera.capture_single_image()
        name=f'{test_camera.collection_name}_{count}'
        time.sleep(1)
        img = cv2.imread(f'./stepper_calibration/{name}.jpg')
        undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)

        try:
            #x,y=eye.get_aruco_coordinates(img, aruco_to_track)
            x,y=eye.get_aruco_coordinates(undistorted, aruco_to_track)

        except ValueError:
            x=None
            y=None

        point_to_transform = np.array([[x,y]], dtype='float32')
        point_to_transform = np.array([point_to_transform])
        #transformed_point = cv2.perspectiveTransform(point_to_transform, dist_matrix)
        transformed_point = cv2.perspectiveTransform(point_to_transform, undist_matrix)
        warp_x, warp_y = transformed_point[0][0]

        point=points[count,:]
        new_point_x,new_point_y=point
        incr_x=new_point_x-prev_point_x
        incr_y=new_point_y-prev_point_y
        pos1,pos2=cm_to_steps(incr_x,incr_y,W,H)
        pos1=int(pos1)
        pos2=int(pos2)
        
        #we are home in the first iteration
        if count == 0:
            prev_x_pix = warp_x
            prev_y_pix = warp_y

        incr_x_pix = warp_x - prev_x_pix
        incr_y_pix = warp_y - prev_y_pix
        dict_to_save['id']=count
        dict_to_save['incr_x']=incr_x_pix
        dict_to_save['incr_y']=incr_y_pix
        dict_to_save['pos1']=pos1
        dict_to_save['pos2']=pos2
        list_of_dict.append(dict_to_save.copy())

        print('Send to arduino:', mode, pos1, pos2)
        stp.sendToArduino(f"{mode},{pos1},{pos2}")

        count+=1
        prev_point_x=new_point_x
        prev_point_y=new_point_y
        prev_x_pix = warp_x
        prev_y_pix = warp_y

#for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf=f'./data/pixel_step.csv', sep=' ',index=False)