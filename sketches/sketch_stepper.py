from pool.stepper import Stepper
from random import randrange
import time

stp=Stepper(baudRate=9600,
            serialPortName='COM3' )

stp.setupSerial()

data = [(-1,0,0),(0,-1800,2700),(0,0,0),(-2,0,0),(-1,0,0),(-2,0,0)]
count=0

data_to_send=data[count]
mode,pos1,pos2=data_to_send
stp.sendToArduino(f"{mode},{pos1},{pos2}")

while True:

    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        
        # when we recieve the data from arduino, we assume the motors have stopped?
        # send a message at intervals
        time.sleep(6)
        count+=1
        data_to_send=data[count]
        #print(count, data,data_to_send)
        mode,pos1,pos2=data_to_send
        print('Send to arduino:', mode, pos1, pos2)
        stp.sendToArduino(f"{mode},{pos1},{pos2}")
