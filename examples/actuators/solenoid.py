from pool.controller_actuators import Controller_actuators
import time

stp=Controller_actuators(baudRate=9600,
            serialPortName='COM3' )

stp.setupSerial()


stp.sendToArduino("-3,0,0")

while True:
    # check for a reply
    arduinoReply = stp.recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        
        # when we recieve the data from arduino, we assume the motors have stopped
        # send a message at intervals
        time.sleep(4)
        mode,pos1,pos2=(-3,0,0)
        print('Send to arduino:', mode, pos1, pos2)
        stp.sendToArduino(f"{mode},{pos1},{pos2}")
        break