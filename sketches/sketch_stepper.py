from pool.stepper import Stepper
from random import randrange
import time

stp=Stepper(baudRate=9600,
            serialPortName='COM3' )

stp.setupSerial()

motor_position = f"2,{randrange(500,1500)},{randrange(500,1500)}"

stp.sendToArduino(motor_position)

while True:
    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        
        # when we recieve the data from arduino, we assume the motors have stopped?
        # send a message at intervals
        time.sleep(10)
        print('send to arduino:')
        stp.sendToArduino(f"2,{randrange(1500,3500)},{randrange(1500,3500)}")
