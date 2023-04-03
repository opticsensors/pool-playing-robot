import serial

from typing import * # type: ignore



class Stepper(object):
    """
    A class that finds the centroid of the pool balls and classifies them

    Attributes
    ----------    
    ...

    """
    def __init__(self, baudRate : int =9600,
                       serialPortName : Optional[str]=None):
        
        if serialPortName is None:
            # get the first port that has "Arduino" in the name
            for port in self._list_com_ports():
                if "Arduino" in port['name']:
                    serialPortName = port['device']
                    break        
        assert serialPortName is not None, "No Arduino found"

        self.startMarker='<'
        self.endMarker='>'
        self.dataStarted=False
        self.dataBuf=""
        self.messageComplete=False
        self.baudRate=baudRate
        self.serialPortName : str =serialPortName

    @staticmethod
    def _list_com_ports():
        com_ports = serial.tools.list_ports.comports()
        port_list = []
        
        for port in com_ports:
            port_info = {
                'device': port.device,
                'name': port.description
            }
            port_list.append(port_info)
        
        return port_list

    def waitForArduino(self):

        # wait until the Arduino sends 'Arduino is ready' - allows time for Arduino reset
        # it also ensures that any bytes left over from a previous message are discarded
        
        print("Waiting for Arduino to reset")
        
        msg = ""
        while msg.find("Arduino is ready") == -1:
            msg = self.recvLikeArduino()
            if not (msg == 'XXX'): 
                print(msg)

    def setupSerial(self):

        self.serialPort = serial.Serial(port= self.serialPortName, baudrate = self.baudRate, timeout=0, rtscts=True)

        print("Serial port " + self.serialPortName + " opened  Baudrate " + str(self.baudRate))

        self.waitForArduino()

    def sendToArduino(self,stringToSend):
        
        stringWithMarkers = (self.startMarker)
        stringWithMarkers += stringToSend
        stringWithMarkers += (self.endMarker)

        self.serialPort.write(stringWithMarkers.encode('utf-8')) # encode needed for Python3

    def recvLikeArduino(self):

        if self.serialPort.inWaiting() > 0 and self.messageComplete == False:
            x = self.serialPort.read().decode("utf-8") # decode needed for Python3
            
            if self.dataStarted == True:
                if x != self.endMarker:
                    self.dataBuf = self.dataBuf + x
                else:
                    self.dataStarted = False
                    self.messageComplete = True
            elif x == self.startMarker:
                self.dataBuf = ''
                self.dataStarted = True
        
        if (self.messageComplete == True):
            self.messageComplete = False
            return self.dataBuf
        else:
            return "XXX" 

