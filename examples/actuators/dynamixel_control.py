from pool.dynamixel import Dynamixel
from random import randrange
import time

dxl=Dynamixel(baudRate=115200,
             serialPortName='com4' )

dxl.setupDynamixel()
dxl.setupPID(P=640,I=800,D=0)

angles=[25,90,135,270]

for angle in angles:
    goal_position = dxl.angle_to_dynamixel_position(angle)
    dxl.sendToDynamixel(goal_position,50, 1)

    while True:
        # Read present position
        present_position = dxl.readDynamixel()
        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (dxl.dynamixel_id, goal_position, present_position))

        if not abs(goal_position - present_position) > dxl.moving_status_threshold:
            break
    time.sleep(7)