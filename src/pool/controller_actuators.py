
from typing import *  # type: ignore

import serial
import serial.tools.list_ports


class Controller_actuators(object):
    """Goes in pair with `serial_control.ino`"""

    def __init__(
        self,
        baudRate       : int           = 9600,
        serialPortName : Optional[str] = None
    ):
        self.connected : bool = True

        if serialPortName is None:
            self.connected : bool = False
            self._automatically_connenct_to_arduino()

        self.startMarker     : str           = '<'
        self.endMarker       : str           = '>'
        self.dataStarted     : bool          = False
        self.dataBuf         : str           = ""
        self.messageComplete : bool          = False
        self.baudRate        : int           = baudRate
        self.serialPortName  : Optional[str] = serialPortName

    def _automatically_connenct_to_arduino(self):
        """Connects to the arduino"""

        # get the first port that has "Arduino" in the name
        for port in self._list_com_ports():
            if "Arduino" in port['name']:
                self.serialPortName = port['device']
                self.connected      = True
                break

    @staticmethod
    def _list_com_ports():
        com_ports = serial.tools.list_ports.comports()
        port_list = []
        
        for port in com_ports:
            port_info = {
                'device': port.device,
                'name'  : port.description
            }
            port_list.append(port_info)
        
        return port_list

    def test_if_connected(self) -> bool:
        """Tests if the arduino is connected"""
        return self.connected

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
        assert self.connected, "Arduino is not connected"

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

