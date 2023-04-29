# To do list



- [ ] Documentation
  - [ ] Document the behaviour of the H configuration

- [ ] Buy
    - [x] heat shrink
    - [ ] multiple usb ports
- [ ] Hardware
    - [x] create aruco supports attached to end effector and pool table


- [ ] Software

- [x] Make a script to compute the repeatability error of aruco detection (take 5 images without moving anything and calculate the coord pixel error or take images after moving to the same position several times)
- [x] make a process to compute the corners before starting the pixel to step calibration (I should be carefull because if I take the image in the homing position, the moving v slot bar will occlude the arucos from the left side). In other words, picture corners.jpg should be updated in another script or in the same script in case the structure moves
- [ ] try different models to compute the pixel to step and recollect more data
    - [x] Linear reg
    - [ ] Other
- [ ] recompute H and W to make point more centered to pool table







## Software

- [ ] Fix: Possible issue with `C:\Github\pool-playing-robot` when calling `save_folder = "data/photos/",`
