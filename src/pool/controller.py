"""Hosts the Controller class.

This class sets up all the architecture for the pool program to run. The main.py
script should isntante this and configure it. In terms of software architecture,
it's intended to be used as a high level class for the robot.
"""

from rich import print

from pool.cam import Camera_DLSR
from pool.eye import Eye
from pool.controller_actuators import Controller_actuators


class Controller:

    def __init__(self) -> None:
        
        self.camera_main : Camera_DLSR = Camera_DLSR(
            image_type      = "jpg",
            collection_name = "poolrobot",
            save_folder     = "data/photos/",
        )
        self.eye : Eye = Eye()
        self.stepper : Controller_actuators = Controller_actuators(9600)

    def check_status(self) -> bool:
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

if __name__ == "__main__":

    c = Controller()
    c.check_status()
