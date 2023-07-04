from pool.cam import Camera_DLSR, Camera_DLSR_settings
import keyboard
import time

camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'

test_camera = Camera_DLSR(control_cmd_location=camera_control_cmd_path, image_type='jpg')
test_setting: Camera_DLSR_settings = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
test_camera.save_folder='./results/'
test_camera.collection_name = 'img'


while True:
    # do something
    time.sleep(0.15)
    if keyboard.is_pressed("q"):
        print("q pressed, ending loop")
        break
    elif keyboard.is_pressed("s"):
        print("s pressed, taking photo")
        test_camera.capture_single_image()