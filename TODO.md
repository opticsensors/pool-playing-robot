# To do list



## Hardware

- [ ] Document
  - [ ] ball sizes + weight
  - [ ] measures of the pool table
  - [ ] Document the behaviour of the H configuration (the drawing of the directions motors needs to scroll to advance or not the carriage)
- [ ] Buy/get
  - [ ] Logic mosfet
  - [x] Beefy solenoid pls
  - [ ] Mini step motor?
  - [ ] MultiUSB port adaptor
- [ ] Actuator 0
  - [ ] First CAD design
- [ ] Actuator 1 (spring loded)
  - [ ] First CAD design



## Software

- [ ] Fix: Possible issue with `C:\Github\pool-playing-robot` when calling `save_folder = "data/photos/",` if CWD is different
- [ ] Refactor
  - [x] Rename `Camera` to `Camera_DLSR`
  - [x] Rename `Camera_settings` to `Camera_DLSR_settings`
- [ ] UX
  - [x] Add UX to camera when being used from main.py
  - [x] Add UX to arduinos  detect them
  - [ ] Add UX to arduinos ping them
- [ ] Arduino controller
  - [x] Safety system with detectors in the limtis of the frame
  - [ ] Change `recvWithStartEndMarkers` so that besides receiving messages like `<X,Y,Z>` it also can receive messages like `/ping/`
- [x] FOV deformation
  - [x] First implementation
  - [x] Execute the implementation
  - [x] Check the roundness of the balls and the distance in between the centroids (measured error might be around 1 or 3 milimeters)
- [ ] Ball detection
  - [x] First YOLO detection
  - [x] Test first YOLO detection
  - [ ] Add UX to YOLO (if file missing or other stuff)
- [ ] Code main controller
  - [x] Make a first iteration
  - [ ] Main controller has working loop to control stuff
  - [ ] Update main to get correct corners etc
  - [ ] Once done, get `main.py` and instantiate the code currently in `Controller.py`
- [ ] Physics simulation
  - [ ] Make first proposal
- [ ] Player algorithm
  - [ ] Make first proposal

















