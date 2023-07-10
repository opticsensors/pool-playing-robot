import time
import datetime
import keyboard
import cv2
from pool.controller_actuators import Controller_actuators
from pool.cam import Camera_DLSR, Camera_DLSR_settings
from pool.calibration import InverseKinematics
from pool.dynamixel import Dynamixel
from pool.ball_detection import Yolo
from pool.brain import ShotSelection
from pool.eye import Eye

print("Done importing!")
date_name = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
f = open(f"./results/LOG_{date_name}.txt", 'w')

#stepper motor initialization
stp=Controller_actuators(baudRate=9600,serialPortName='COM3')
stp.setupSerial()

#dynamixel initialization
dxl=Dynamixel(baudRate=115200,serialPortName='com4')
dxl.setupDynamixel()
dxl.setupPID(P=640,I=800,D=0)

#camera initialization
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
camera = Camera_DLSR(control_cmd_location=camera_control_cmd_path, image_type='jpg')
camera_setting = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
camera.save_folder='./results/'
camera.collection_name = f'{date_name}'

#necessary objects
ik = InverseKinematics()
yolo = Yolo()
ss = ShotSelection()
eye = Eye()

#init position and stepper mode
mode = 0
rotated=False
count_iter=0
turn='solid'
activated = True

#go home
stp.sendToArduino(f"-1,0,0")

while activated:
    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Arduino reply: ", arduinoReply)

        if count_iter==0: # we are at home position
            print('Waiting for user input to take photo and compute cue ball centroid and angle')
            while True:
                time.sleep(0.15)
                if keyboard.is_pressed("c"):
                    print("c pressed, taking photo and calculating ...")
                    camera.capture_single_image() 
                    time.sleep(0.5)
                    img = cv2.imread('./results/img_0.jpg')
                    img_undist_warp = eye.undistort_and_warp_image(img)
                    d_centroids, _ = yolo.detect_balls(img_undist_warp,conf=0.25, overlap_threshold=100)
                    uvBallCentroid = d_centroids[0]
                    angle = ss.get_actuator_angle(d_centroids, turn) #  returns angles in the range [-180,180]
                    angle_dxl = angle + 90 
                    print("angle: ", angle, file=f)
                    print("angle_dxl: ", angle_dxl, file=f)
                    print("d_centroids: ", d_centroids, file=f)
                    print("uvBallCentroid: ", uvBallCentroid, file=f)
                    time.sleep(0.5)
                    break

            steps1,steps2=ik.img_data_to_steps(uvBallCentroid, angle)
            steps1=int(steps1)
            steps2=int(steps2)
            print("steps: ", steps1, steps2, file=f)
            print('Send to arduino:', mode, steps1, steps2)
            stp.sendToArduino(f"{mode},{steps1},{steps2}")
            rotated=False
            count_iter+=1

        else:
            print('Waiting for user input to rotate flipper and activate solenoid')
            while True:
                time.sleep(0.15)
                if keyboard.is_pressed("s") and rotated:
                    print("s pressed, activating solenoid")
                    time.sleep(0.5)
                    stp.sendToArduino(f"100,0,0")
                    activated = False
                    break
                if keyboard.is_pressed("r") and not rotated:
                    print("r pressed, rotating end effector")
                    goal_position = dxl.angle_to_dynamixel_position(angle_dxl)
                    dxl.sendToDynamixel(int(goal_position),50, 1)
                    print("goal_position: ", goal_position, file=f)
                    for i in range(10,0,-1):
                        print("Rotating... time remaining: {:2d}s".format(i), end="\r", flush=True)
                        time.sleep(1)
                    present_position = dxl.readDynamixel()
                    print("present_position: ", present_position, file=f)
                    rotated=True
                if keyboard.is_pressed("q"):
                    print("q pressed, exiting program")
                    activated = False
                    f.close()
                    break

f.close()