import time
import pandas as pd
from pool.controller_actuators import Controller_actuators
from pool.cam import Camera_DLSR, Camera_DLSR_settings
from pool.calibration import InverseKinematics

#stepper motor initialization
stp=Controller_actuators(baudRate=9600, serialPortName='COM3')
stp.setupSerial()

#camera initialization
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
camera = Camera_DLSR(control_cmd_location=camera_control_cmd_path, image_type='jpg')
camera_setting = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
camera.save_folder='./results/'
camera.collection_name = 'img'

#helper functions for calibration
ik = InverseKinematics()

#define calibration points
points=ik.generate_calibration_points((3,2))
print(points)

#init position and stepper mode
mode = 0
prev_point_x=0
prev_point_y=0
dict_to_save = {}
list_of_dict = []
img_num=1

#go home
stp.sendToArduino(f"-1,0,0")
dict_to_save={'point_x':0,'point_y':0,'incr_x':0, 'incr_y':0,'steps1':0,'steps2':0,'img_num':0, 'img_name':'img_0'}
list_of_dict.append(dict_to_save.copy())

for point in points:
    new_point_x,new_point_y=point
    print(point)
    while True:
        # check for a reply
        arduinoReply = stp.recvLikeArduino()
        if not (arduinoReply == 'XXX'):
            print ("Reply: ", arduinoReply)
            time.sleep(1)
            camera.capture_single_image() # image is taken when carriage is in prev_point (not new_point) -> first image is in home position!
            time.sleep(1)
            incr_x=new_point_x-prev_point_x
            incr_y=new_point_y-prev_point_y
            steps1,steps2=ik.cm_to_steps(incr_x,incr_y)
            steps1=int(steps1)
            steps2=int(steps2)
            print('Send to arduino:', mode, steps1, steps2)
            stp.sendToArduino(f"{mode},{steps1},{steps2}")

            # save data of new_point (before picture is taken)
            dict_to_save['point_x']=new_point_x
            dict_to_save['point_y']=new_point_y
            dict_to_save['incr_x']=incr_x
            dict_to_save['incr_y']=incr_y
            dict_to_save['steps1']=steps1
            dict_to_save['steps2']=steps2
            dict_to_save['img_num']=img_num
            list_of_dict.append(dict_to_save.copy())
            
            prev_point_x = new_point_x
            prev_point_y = new_point_y
            img_num+=1
            break
time.sleep(1)
camera.capture_single_image() # capture last image
time.sleep(1)
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf='./results/calibration_points.csv', sep=',',index=False)