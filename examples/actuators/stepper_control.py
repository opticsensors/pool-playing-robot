from pool.controller_actuators import Controller_actuators
import time

stp=Controller_actuators(baudRate=9600,
            serialPortName='COM3' )

stp.setupSerial()

data = [(-1,0,0),(0,0,700),(0,0,500)]

stp.sendToArduino("-1,0,0")
print('home yet?')
for data_to_send in data:
    print(data)
    while True:
        # check for a reply
        arduinoReply = stp.recvLikeArduino()
        if not (arduinoReply == 'XXX'):
            print ("Reply: ", arduinoReply)
            
            # when we recieve the data from arduino, we assume the motors have stopped
            # send a message at intervals
            time.sleep(4)
            mode,pos1,pos2=data_to_send
            print('Send to arduino:', mode, pos1, pos2)
            stp.sendToArduino(f"{mode},{pos1},{pos2}")
            break