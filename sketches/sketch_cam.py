from pool.cam import Camera
import time

camera_control_cmd_path = 'C:\\Program Files (x86)\\digiCamControl\\CameraControlCmd.exe'

test_camera = Camera(control_cmd_location=camera_control_cmd_path)
test_setting: Camera.Settings = Camera.Settings(aperture='4.5', shutter_speed='1/60', iso='200')
test_camera.save_folder='./results/'
test_camera.collection_name = 'img'

start=time.time()
test_camera.capture_single_image()
end=time.time()

print(end-start)
