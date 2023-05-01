import time
import numpy as np
from pool.stepper import Controller_actuators
from pool.cam import Camera, Camera_settings
import time


#stepper motor initialization
stp=Controller_actuators(baudRate=9600,serialPortName='COM3' )
stp.setupSerial()

#camera initialization
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
test_camera = Camera(control_cmd_location=camera_control_cmd_path)
test_setting: Camera_settings = Camera_settings(aperture='4', shutter_speed='1/10', iso='400')
test_camera.save_folder='./stepper_repeatability/'
test_camera.collection_name = 'img'

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

#define needed variables
repeatability=3
points=generate_grid(2,2)
np.random.shuffle(points) 
repe_points=points.copy()
for _ in range(repeatability):
    repe_points = np.vstack([repe_points, points])
np.save('./data/repeatability_points.npy', repe_points)
#homig_position = [0,0]
#repe_points = np.vstack([homig_position, repe_points])
print(repe_points)
mode = 0
count = 0
pos1 = 0
pos2 = 0
prev_point_x=0
prev_point_y=0
W = 84
H = 44
stp.sendToArduino(f"-1,0,0")

while True:
    if count>repe_points.shape[0]:
        break

    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        time.sleep(1)
        test_camera.capture_single_image()
        time.sleep(1)
        point=repe_points[count,:]
        new_point_x,new_point_y=point
        incr_x=new_point_x-prev_point_x
        incr_y=new_point_y-prev_point_y
        pos1,pos2=cm_to_steps(incr_x,incr_y,W,H)
        pos1=int(pos1)
        pos2=int(pos2)
        print('Send to arduino:', mode, pos1, pos2)
        stp.sendToArduino(f"{mode},{pos1},{pos2}")
        count+=1
        prev_point_x = new_point_x
        prev_point_y = new_point_y