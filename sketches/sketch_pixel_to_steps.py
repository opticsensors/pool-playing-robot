from pool.cam import Camera
import time
import cv2
from pool.eye import Eye
import numpy as np
from pool.stepper import Stepper
from random import randrange
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

def interpolation_steps(point, max_phi1, max_phi2):
    alpha, beta = point
    phi1 = 0.5 * (beta*(max_phi1+max_phi2)-alpha*(-max_phi1+max_phi2))
    phi2 = 0.5 * (beta*(max_phi1+max_phi2)+alpha*(-max_phi1+max_phi2))
    return phi1,phi2


#############################################
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
test_camera = Camera(control_cmd_location=camera_control_cmd_path)
test_setting: Camera.Settings = Camera.Settings(aperture='4', shutter_speed='1/10', iso='400')
test_camera.save_folder='./results/'
test_camera.collection_name = 'img'

##############################################
eye=Eye()
eye.bottom_aruco_ids=[17,8]
eye.top_aruco_ids=[9,3]
eye.left_aruco_ids=[1,11]
eye.right_aruco_ids=[23,10]

###############################################

stp=Stepper(baudRate=9600,serialPortName='COM3' )
stp.setupSerial()

def get_image(count):
    test_camera.capture_single_image()
    name=f'{test_camera.collection_name}_{count}'
    img = cv2.imread(f'./data/{name}.jpg')
    cameraMatrix=np.load('./data/cameraMatrix.npy')
    dist=np.load('./data/dist.npy')

    undistorted=eye.undistort_image(img, cameraMatrix, dist, remapping=False)
    corners=eye.get_pool_corners(undistorted)
    warp=eye.perspective_transform(undistorted,corners)
    return warp

points=generate_grid(2,2)
np.random.shuffle(points) 
mode=0
count=0
max_phi1 = -4107
max_phi2 = 24263
aruco_to_track=1
stp.sendToArduino(f"-1,0,0")

while True:
    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        time.sleep(3)
        img=get_image(count)
        x,y=eye.get_aruco_coordinates(aruco_to_track)
        point=points[count,:]
        pos1,pos2=interpolation_steps(point,max_phi1,max_phi2)
        
        print('Send to arduino:', mode, pos1, pos2)
        stp.sendToArduino(f"{mode},{pos1},{pos2}")

        count+=1


