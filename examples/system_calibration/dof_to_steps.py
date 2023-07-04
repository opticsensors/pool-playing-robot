import time
import keyboard
import random
from pool.controller_actuators import Controller_actuators
from pool.cam import Camera_DLSR, Camera_DLSR_settings
from pool.calibration import InverseKinematics
from pool.dynamixel import Dynamixel

#stepper motor initialization
stp=Controller_actuators(baudRate=9600,serialPortName='COM3' )
stp.setupSerial()

#dynamixel initialization
dxl=Dynamixel(baudRate=115200,
             serialPortName='com4')
dxl.setupDynamixel()
dxl.setupPID(P=640,I=800,D=0)

#camera initialization
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
camera = Camera_DLSR(control_cmd_location=camera_control_cmd_path, image_type='jpg')
camera_setting = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
camera.save_folder='./results/'
camera.collection_name = 'img'

#helper functions for calibration
ik = InverseKinematics()

#define calibration points
points=ik.generate_and_save_calibration_points((6,4),name='system_calibration_points', reduce=(0.7,0.7), random_order=False)
print(points)

#init position and stepper mode
mode = 0
prev_point_x=0
prev_point_y=0
rotated=False

#go home
stp.sendToArduino(f"-1,0,0")

for point in points:
    new_point_x,new_point_y=point

    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)

        while True:
            time.sleep(0.15)
            if keyboard.is_pressed("i"):
                print("i pressed, taking photo")
                camera.capture_single_image() 
                time.sleep(0.5)
                break
            if keyboard.is_pressed("r") and not rotated:
                print("r pressed, rotating end effector")
                goal_position = dxl.angle_to_dynamixel_position(round(random.uniform(0,360), 4))
                dxl.sendToDynamixel(goal_position,50, 1)
                time.sleep(10) # after 10 seconds we will have reached goal pos
                present_position = dxl.readDynamixel()
                rotated=True
            
        incr_x=new_point_x-prev_point_x
        incr_y=new_point_y-prev_point_y
        steps1,steps2=ik.cm_to_steps(incr_x,incr_y)
        steps1=int(steps1)
        steps2=int(steps2)
        print('Send to arduino:', mode, steps1, steps2)
        stp.sendToArduino(f"{mode},{steps1},{steps2}")
        prev_point_x = new_point_x
        prev_point_y = new_point_y
        rotated=False