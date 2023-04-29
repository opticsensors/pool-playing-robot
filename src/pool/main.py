

from pool.stepper import Stepper
from pool.dynamixel import Dynamixel
from pool.eye import Eye
from pool.cam import Camera, Camera_settings
import time
import cv2
import numpy as np



def main():

    # Camera
    # settings: Camera_settings = Camera_settings(
    #     aperture      = '4.5',
    #     shutter_speed = '1/60',
    #     iso           = '200',
    # )
    camera : Camera = Camera(
        image_type      = "jpg",
        collection_name = "poolrobot",
        save_folder     = "data/photos/",
    )

    # camera.test()

    # Eye
    eye = Eye()

    # # Stepper
    # stp = Stepper(baudRate=9600, serialPortName='COM3')
    # stp.setupSerial()
    # # Servo
    # dxl = Dynamixel(baudRate=115200, serialPortName='com4')
    # dxl.setupDynamixel()




    # get corners (we will use this corners from now on)
    camera.capture_single_image()



    img = cv2.imread('./results/img1.jpg')
    corners=eye.get_pool_corners(img)
    # more operations like:
    # - parameter tuning
    # - warp background
    # - ...

    #######################


    while True:
        camera.capture_single_image()
        #read image 
        img = cv2.imread('./results/img.jpg')
        #trasnform image
        warp=eye.perspective_transform(img,corners)
        # Yolov8 results
        d_centroids=eye.YOLO(warp,conf=0.25, overlap_threshold=100)

    



if __name__ == "__main__":

    main()
