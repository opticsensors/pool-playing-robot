from pool.stepper import Stepper
from random import randrange
import time

stp=Stepper(baudRate=9600,
            serialPortName='COM3' )

stp.setupSerial()

data = [(0,0,1500),(0,-500,500), (0,700,-700)]
count=0

data_to_send=data[count]
mode,pos1,pos2=data_to_send
stp.sendToArduino(f"{mode},{pos1},{pos2}")

while True:
    count+=1
    data_to_send=data[count]
    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        
        # when we recieve the data from arduino, we assume the motors have stopped?
        # send a message at intervals
        time.sleep(10)
        print('send to arduino:')
        stp.sendToArduino(f"{mode},{pos1},{pos2}")
