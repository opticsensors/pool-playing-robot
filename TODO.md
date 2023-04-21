# To do list
- Buy the following parts:
    - heat shrink
    - multiple usb ports
- create aruco supports attached to end effector and pool table
- aruco detection is not accurate after camera undistortion (should I apply undistort transformation after the aruco coordinates have been detected in the distorted image?)
- make a script to compute the repeatability error of aruco detection (take 5 images without moving anything and calculate the coord pixel error)
- make a process to compute the corners before starting the pixel to step calibration (I should be carefull because if I take the image in the homing position, the moving v slot bar will occlude the arucos from the left side)


