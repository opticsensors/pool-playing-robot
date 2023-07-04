import time
import keyboard
from pool.controller_actuators import Controller_actuators
from pool.dynamixel import Dynamixel

#stepper motor initialization
stp=Controller_actuators(baudRate=9600,serialPortName='COM3')
stp.setupSerial()

#dynamixel initialization
dxl=Dynamixel(baudRate=115200,
             serialPortName='com4')
dxl.setupDynamixel()
dxl.setupPID(P=640,I=800,D=0)

steps = [(0,0,700),(0,0,800)]
angles=[0,45]
rotated=False

#go home
stp.sendToArduino("-1,0,0")

for step,angle in zip(steps,angles):

    while True:
        # check for a reply
        arduinoReply = stp.recvLikeArduino()
        if not (arduinoReply == 'XXX'):
            print ("Reply: ", arduinoReply)

            while True:
                time.sleep(0.15)
                if keyboard.is_pressed("r") and not rotated:
                    print("r pressed, rotating end effector")
                    goal_position = dxl.angle_to_dynamixel_position(angle)
                    dxl.sendToDynamixel(goal_position,50, 1)
                    time.sleep(10) # after 10 seconds we will have reached goal pos
                    present_position = dxl.readDynamixel()
                    rotated=True
                    break                
            mode,pos1,pos2=step
            print('Send to arduino:', mode, pos1, pos2)
            stp.sendToArduino(f"{mode},{pos1},{pos2}")
            rotated=False
            break

dxl.closeDynamixel()