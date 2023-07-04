import time
import keyboard
import random
import pandas as pd
from pool.controller_actuators import Controller_actuators
from pool.cam import Camera_DLSR, Camera_DLSR_settings
from pool.calibration import InverseKinematics
from pool.dynamixel import Dynamixel

#stepper motor initialization
stp=Controller_actuators(baudRate=9600,serialPortName='COM3')
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
points=ik.generate_calibration_points((4,3), reduce=(0.9,0.5), random_order=False)
print(points)

#init position and stepper mode
mode = 0
prev_point_x=0
prev_point_y=0
rotated=False
img_num=1
dict_to_save = {}
list_of_dict = []

#go home
stp.sendToArduino(f"-1,0,0")

for point in points:
    new_point_x,new_point_y=point

    while True:
        # check for a reply
        arduinoReply = stp.recvLikeArduino()
        if not (arduinoReply == 'XXX'):
            print ("Reply: ", arduinoReply)

            while True:
                time.sleep(0.15)
                if keyboard.is_pressed("i") and rotated:
                    print("i pressed, taking photo")
                    camera.capture_single_image() 
                    time.sleep(0.5)
                    break
                if keyboard.is_pressed("r") and not rotated:
                    print("r pressed, rotating end effector")
                    angle=round(random.uniform(0,360), 4)
                    goal_position = dxl.angle_to_dynamixel_position(angle)
                    dxl.sendToDynamixel(goal_position,50, 1)
                    time.sleep(7) # after 7 seconds we will have reached goal pos
                    present_position = dxl.readDynamixel()
                    rotated=True
                
            incr_x=new_point_x-prev_point_x
            incr_y=new_point_y-prev_point_y
            steps1,steps2=ik.cm_to_steps(incr_x,incr_y)
            steps1=int(steps1)
            steps2=int(steps2)
            print('Send to arduino:', mode, steps1, steps2)
            stp.sendToArduino(f"{mode},{steps1},{steps2}")

            dict_to_save['point_x']=new_point_x
            dict_to_save['point_y']=new_point_y
            dict_to_save['incr_x']=incr_x
            dict_to_save['incr_y']=incr_y
            dict_to_save['steps1']=steps1
            dict_to_save['steps2']=steps2
            dict_to_save['angle']=angle
            dict_to_save['img_num']=img_num
            list_of_dict.append(dict_to_save.copy())

            prev_point_x = new_point_x
            prev_point_y = new_point_y
            rotated=False
            img_num+=1
            break

df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf='./results/calibration_image_data.csv', sep=',',index=False)