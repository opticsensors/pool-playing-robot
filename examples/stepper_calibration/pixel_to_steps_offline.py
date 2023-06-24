import time
import numpy as np
from pool.controller_actuators import Controller_actuators
from pool.cam import Camera_DLSR, Camera_DLSR_settings
from pool.pixel_to_steps import PixelSteps
import time


#stepper motor initialization
stp=Controller_actuators(baudRate=9600,serialPortName='COM3' )
stp.setupSerial()

#camera initialization
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
camera = Camera_DLSR(control_cmd_location=camera_control_cmd_path, image_type='jpg')
camera_setting = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
camera.save_folder='./results/'
camera.collection_name = 'img'

#helper functions for calibration
ps = PixelSteps()

#define calibration points
points=ps.generate_and_save_calibration_points((6,4))
print(points)

#init position and stepper mode
mode = 0
prev_point_x=0
prev_point_y=0

#go home
stp.sendToArduino(f"-1,0,0")

for point in points:
    new_point_x,new_point_y=point

    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        time.sleep(1)
        camera.capture_single_image() #TODO first image will be in home position (not encoded in points!!), make it clearer?
        time.sleep(1)
        incr_x=new_point_x-prev_point_x
        incr_y=new_point_y-prev_point_y
        steps1,steps2=ps.cm_to_steps(incr_x,incr_y)
        steps1=int(steps1)
        steps2=int(steps2)
        print('Send to arduino:', mode, steps1, steps2)
        stp.sendToArduino(f"{mode},{steps1},{steps2}")
        prev_point_x = new_point_x
        prev_point_y = new_point_y