# pip install dynamixel-sdk
from dynamixel_sdk import * # Uses Dynamixel SDK library

class Dynamixel(object):
    """
    A class that ...

    Attributes
    ----------    
    ...

    """
    def __init__(self, baudRate=115200,
                       serialPortName='com4' ):
        
        self.dynamixel_name              = 'X_SERIES'
        self.addr_P                      = 84
        self.addr_I                      = 82
        self.addr_D                      = 80
        self.addr_torque_enable          = 64
        self.addr_goal_position          = 116
        self.addr_profile_velocity       = 112
        self.addr_profile_acceleration   = 108
        self.addr_present_position       = 132
        self.min_position                = 0         
        self.max_position                = 4095      
        self.protocol_version            = 2.0
        self.dynamixel_id                = 1
        self.torque_enable               = 1     # Value for enabling the torque
        self.torque_disable              = 0     # Value for disabling the torque
        self.moving_status_threshold     = 20    # Dynamixel moving status threshold
        self.baudRate                    = baudRate
        self.serialPortName              = serialPortName

    def setupDynamixel(self):

        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.portHandler = PortHandler(self.serialPortName)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(self.protocol_version)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")

        # Set port baudrate
        if self.portHandler.setBaudRate(self.baudRate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.dynamixel_id,
                                                                  self.addr_torque_enable, self.torque_enable)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")

    def closeDynamixel(self):
        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, 
                                                                       self.dynamixel_id, 
                                                                       self.addr_torque_enable, 
                                                                       self.torque_disable)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        # Close port
        self.portHandler.closePort()

    def setupPID(self, P,I,D):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, 
                                                                       self.dynamixel_id, 
                                                                       self.addr_P, 
                                                                       P)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, 
                                                                       self.dynamixel_id, 
                                                                       self.addr_I, 
                                                                       I)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, 
                                                                       self.dynamixel_id, 
                                                                       self.addr_D, 
                                                                       D)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def sendToDynamixel(self,goal_position, velocity, acceleration):
        
        # goal position must be between min and max positions
        # clamping
        goal_position=max(self.min_position, min(goal_position, self.max_position))

        # Write profile velocity
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, 
                                                                       self.dynamixel_id, 
                                                                       self.addr_profile_velocity, 
                                                                       velocity)
        # Write profile accel
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, 
                                                                       self.dynamixel_id, 
                                                                       self.addr_profile_acceleration, 
                                                                       acceleration)
        # Write goal position
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, 
                                                                       self.dynamixel_id, 
                                                                       self.addr_goal_position, 
                                                                       goal_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def readDynamixel(self):

        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, 
                                                                                            self.dynamixel_id, 
                                                                                            self.addr_present_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        return dxl_present_position
    
    def angle_to_dynamixel_position(self, angle): # angle in degrees!
        return (angle/360)*(self.max_position-self.min_position)+self.min_position
