from pool.stepper import Stepper
from pool.dynamixel import Dynamixel
from pool.eye import Eye
from pool.cam import Camera
import time
import cv2
import numpy as np

#######################
# camera initialization
camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
camera = Camera(control_cmd_location=camera_control_cmd_path)
setting: Camera.Settings = Camera.Settings(aperture='4.5', shutter_speed='1/60', iso='200')
camera.save_folder='./results/'
camera.collection_name = 'img'

#######################
# computer vision initialization
eye=Eye()
# get corners (we will use this corners from now on)
camera.capture_single_image()
img = cv2.imread('./results/img1.jpg')
corners=eye.get_pool_corners(img)
# more operations like:
# - parameter tuning
# - warp background
# - ...

#######################
#motor initialization
stp=Stepper(baudRate=9600,
            serialPortName='COM3' )
stp.setupSerial()
dxl=Dynamixel(baudRate=115200,
             serialPortName='com4' )
dxl.setupDynamixel()

while True:
    camera.capture_single_image()
    #read image 
    img = cv2.imread('./results/img.jpg')
    #trasnform image
    warp=eye.perspective_transform(img,corners)
    # Yolov8 results
    d_centroids=eye.YOLO(warp,conf=0.25, overlap_threshold=100)

    


