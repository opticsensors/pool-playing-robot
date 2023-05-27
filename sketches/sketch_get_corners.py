from pool.cam import Camera_DLSR, Camera_DLSR_settings
import keyboard
import time

camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'

test_setting : Camera_DLSR_settings = Camera_DLSR_settings(aperture='4', shutter_speed='1/10', iso='400')
camera = Camera_DLSR()

assert camera.test_if_any_cameras_are_connected(), "No camera detected"

camera.save_folder     = './data/'
camera.collection_name = 'corners'

print ("Press 'q' to quit, 's' to take a photo...")

while True:
    # do something
    time.sleep(0.15)
    if keyboard.is_pressed("q"):
        print("q pressed, ending loop")
        break
    elif keyboard.is_pressed("s"):
        print("s pressed, taking photo")
        camera.capture_single_image()
