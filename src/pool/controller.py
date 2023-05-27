"""Hosts the Controller class.

This class sets up all the architecture for the pool program to run. The main.py
script should isntante this and configure it. In terms of software architecture,
it's intended to be used as a high level class for the robot.
"""

import cv2
import numpy as np
from rich import print

from pool.cam import Camera_DLSR
from pool.controller_actuators import Controller_actuators
from pool.eye import Eye


class Controller:

    def __init__(self) -> None:
        
        # Camera
        self.camera_main : Camera_DLSR = Camera_DLSR(
            image_type      = "jpg",
            collection_name = "poolrobot",
            save_folder     = "data/photos/",
        )

        # Eye
        self.eye : Eye = Eye()

        # Controller
        self.stepper : Controller_actuators = Controller_actuators(9600)

    def check_status_is_good(self) -> bool:
        """Makes sure all the parts are connected etc"""

        to_return : bool = True

        # We check the camera
        o = self.camera_main.test_if_any_cameras_are_connected()
        if not o: print("[red]No camera has been detected...");to_return = False
        else:     print("[green]Camera ok!")

        # We check the arduino controller
        o = self.stepper.test_if_connected()
        if not o: print("[red]No arduino has been detected...");to_return = False
        else:     print("[green]Arduino ok!")

        return to_return
    
    def run(self):

        counter_iteration : int = 0
        while True:

            print(f"Counter_iteration: {counter_iteration}")

            # We take a photo
            c.camera_main.capture_single_image()

            # We undistort the image
            img            = cv2.imread(f'./data/photos/poolrobot_0.jpg')
            cameraMatrix   = np.load('./sketches/data/cameraMatrix.npy')
            dist           = np.load('./sketches/data/dist.npy')
            undist_img     = self.eye.undistort_image(
                img,
                cameraMatrix,
                dist,
                remapping=False,
            )
            undist_corners = self.eye.get_pool_corners(
                undist_img,
                bottom_aruco_ids = [6,7],
                top_aruco_ids    = [2,3],
                left_aruco_ids   = [0,1],
                right_aruco_ids  = [4,5],
            )
            undistorted    = self.eye.undistort_image(img, cameraMatrix, dist, remapping=False)
            a                = self.eye.calculate_perspective_matrix(undist_corners)
            warp_undistorted = self.eye.transform_image_given_a_matrix(
                undistorted,
                undist_corners,
                a,
            )
            cv2.imwrite('./data/corners_0_clean.jpg',warp_undistorted)

            # Apply YOLO
            d_centroids, _ = self.eye.YOLO(
                warp_undistorted,
                conf              = 0.25,
                overlap_threshold = 100,
                data_path         = './sketches/data/data.yaml',
                model_path        = './sketches/data/yolov8m_version9.pt',
            )
            for ball_num in d_centroids:
                x,y=d_centroids[ball_num]
                warp_undistorted=cv2.putText(warp_undistorted, "#{}".format(ball_num), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                warp_undistorted=cv2.circle(warp_undistorted, (int(x), int(y)), 8, (255, 0, 255), -1)
            cv2.imwrite('./data/corners_0_clean_detected.jpg', warp_undistorted)

            p = 0

if __name__ == "__main__":

    c = Controller()
    if c.check_status_is_good():
        c.run()
    c.run()


