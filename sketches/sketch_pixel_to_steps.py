from pool.cam import Camera
import time
import cv2
from pool.eye import Eye
import numpy as np
from pool.stepper import Stepper
import pandas as pd
import time

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


#############################################
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
test_camera = Camera(control_cmd_location=camera_control_cmd_path)
test_setting: Camera.Settings = Camera.Settings(aperture='4', shutter_speed='1/10', iso='400')
test_camera.save_folder='./results/'
test_camera.collection_name = 'img'

##############################################
eye=Eye()

###############################################

stp=Stepper(baudRate=9600,serialPortName='COM3' )
stp.setupSerial()

def get_undistorted_warp_image(count, cameraMatrix, dist, corners):
    test_camera.capture_single_image()
    name=f'{test_camera.collection_name}_{count}'
    img = cv2.imread(f'./results/{name}.jpg')
    undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
    warp=eye.perspective_transform(undistorted,corners)
    return warp

img=cv2.imread(f'./results/corners.jpg')
corners=eye.get_pool_corners(img, bottom_aruco_ids=[9,10,11,0],top_aruco_ids=[3,4,5,6],left_aruco_ids=[1,2],right_aruco_ids=[7,8])
cameraMatrix=np.load('./data/cameraMatrix.npy')
dist=np.load('./data/dist.npy')
points=generate_grid(2,2)
np.random.shuffle(points) 
mode=0
count=0
dict_to_save={}
list_of_dict=[]
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
        time.sleep(3)

        img=get_undistorted_warp_image(count,cameraMatrix, dist, corners)
        try:
            x,y=eye.get_aruco_coordinates(img, aruco_to_track)

        except ValueError:
            x=None
            y=None

        dict_to_save['id']=count
        dict_to_save['x']=x
        dict_to_save['y']=y
        dict_to_save['pos1']=pos1
        dict_to_save['pos2']=pos2
        list_of_dict.append(dict_to_save.copy())

        point=points[count,:]
        new_point_x,new_point_y=point
        incr_x=new_point_x-prev_point_x
        incr_y=new_point_y-prev_point_y
        pos1,pos2=cm_to_steps(incr_x,incr_y,W,H)
        print('Send to arduino:', mode, pos1, pos2)
        stp.sendToArduino(f"{mode},{pos1},{pos2}")

        count+=1
        prev_point_x=new_point_x
        prev_point_y=new_point_y

#for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf=f'./data/pixel_step.csv', sep=' ',index=False)