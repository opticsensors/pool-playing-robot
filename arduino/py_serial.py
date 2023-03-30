import serial
import time
from random import randrange

startMarker = '<'
endMarker = '>'
dataStarted = False
dataBuf = ""
messageComplete = False
position_achieved=True

#========================
#========================
    # the functions

def setupSerial(baudRate, serialPortName):
    
    global  serialPort
    
    serialPort = serial.Serial(port= serialPortName, baudrate = baudRate, timeout=0, rtscts=True)

    print("Serial port " + serialPortName + " opened  Baudrate " + str(baudRate))

    waitForArduino()

#========================

def sendToArduino(stringToSend):
    
        # this adds the start- and end-markers before sending
    global startMarker, endMarker, serialPort
    
    stringWithMarkers = (startMarker)
    stringWithMarkers += stringToSend
    stringWithMarkers += (endMarker)

    serialPort.write(stringWithMarkers.encode('utf-8')) # encode needed for Python3


#==================

def recvLikeArduino():

    global startMarker, endMarker, serialPort, dataStarted, dataBuf, messageComplete

    if serialPort.inWaiting() > 0 and messageComplete == False:
        x = serialPort.read().decode("utf-8") # decode needed for Python3
        
        if dataStarted == True:
            if x != endMarker:
                dataBuf = dataBuf + x
            else:
                dataStarted = False
                messageComplete = True
        elif x == startMarker:
            dataBuf = ''
            dataStarted = True
    
    if (messageComplete == True):
        messageComplete = False
        return dataBuf
    else:
        return "XXX" 

#==================

def waitForArduino():

    # wait until the Arduino sends 'Arduino is ready' - allows time for Arduino reset
    # it also ensures that any bytes left over from a previous message are discarded
    
    print("Waiting for Arduino to reset")
     
    msg = ""
    while msg.find("Arduino is ready") == -1:
        msg = recvLikeArduino()
        if not (msg == 'XXX'): 
            print(msg)



#====================
#====================
    # the program


setupSerial(9600, "COM3")
count = 0
sendToArduino(f"{randrange(500,1500)},{randrange(500,1500)}")
while True:
            # check for a reply
    arduinoReply = recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print ("Reply: ", arduinoReply)
        # when we recieve the data from arduino, we assume the motors have stopped?
        # position_achieved=True # this bool var is the same as messageComplete
        
        # send a message at intervals
        time.sleep(10)
        print('send to arduino:')
        sendToArduino(f"{randrange(500,1500)},{randrange(500,1500)}")
        count += 1
        # position_achieved=False
