from pool.stepper import Stepper
from pool.dynamixel import Dynamixel
from pool.eye import Eye
from pool.cam import Camera
import time
import cv2
import numpy as np


camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
camera = Camera(control_cmd_location=camera_control_cmd_path)
setting: Camera.Settings = Camera.Settings(aperture='4.5', shutter_speed='1/60', iso='200')
camera.save_folder='./results/'
camera.collection_name = 'img'

eye=Eye()
bg = cv2.imread('./data/background.jpg')
camera.capture_single_image()
#read image and background
img = cv2.imread('./results/img.jpg')
#get corners: top-right, top-left, bottom-left, bottom-right
corners=eye.get_pool_corners(img)
warp_bg=eye.perspective_transform(bg,corners)

while True:
    camera.capture_single_image()
    #read image and background
    img = cv2.imread('./results/img.jpg')
    warp=eye.perspective_transform(img,corners)

    # Yolov8 results
    d_centroids=eye.YOLO(warp,conf=0.25, overlap_threshold=100)

    


