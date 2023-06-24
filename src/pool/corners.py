import os
import cv2
from pool.cam import Camera_DLSR, Camera_DLSR_settings
from pool.utils import Params

def save_img_corners():

    camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'
    camera = Camera_DLSR(control_cmd_location=camera_control_cmd_path, image_type='jpg')
    setting: Camera_DLSR_settings = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
    camera.save_folder=os.path.join(Params().PATH_REPO, 'data')
    camera.collection_name = 'corners'
    camera.capture_single_image()

def load_img_corners():
    return cv2.imread(os.path.join(Params().PATH_REPO, 'data', 'corners_0.jpg'))