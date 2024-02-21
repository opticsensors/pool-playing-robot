# `pool-paying-robot`

Design and implementation of a robot that plays the game of pool autonomously. To achieve this goal, three different branches had to be explored:
- **Vision system**: hardware and computer vision algorithms used for detecting and identifying pool balls
- **Shot selection**: algorithms to find the *best* shot given the balls configuration and pool table dimensions
- **Actuation system**: how to send commands to the actuators and the mechanical parts to make the execution of pool shots possible

## Getting started

Run the following command for clonning the repository from GitHub:

```shell
git clone https://github.com/opticsensors/pool-playing-robot.git
```

Then:

1. Make sure  you have the latest version of pip and PyPA’s build installed:
   ```shell
   py -m pip install --upgrade pip
   py -m pip install --upgrade build
   ```

2. In your terminal, navigate to your new project, and initialise Git
   ```shell
   git init
   ```

3. Install the necessary packages using `pip`:
   ```shell
   py -m build
   py -m pip install -e .
   ```

4. Stage all your project files, and make your first commit
   ```shell
   git add .
   git commit -m "First commit"
   ```


## Directory structure
The project follows this tree structure:

```
├── README.md                   <- The top-level README for developers using this project.
├── data                        <- Camera matrix, lens parameters, pool table corners, CNN weights ...
├── cad                         <- CAD files of the pool playing robot parts and the pool table. 
├── arduino                     <- Controls two stepper motors for positioning tasks, with options 
│                                  for homing, microstepping, and activating a solenoid.
│               
├── examples                    <- Python scripts.
│               
├── pyproject.toml              <- Makes project pip installable so src can be imported.               
│                                  The requirements are listed here.
│                
├── src                         <- Source code of this project.
│   └── pool            
│       └── __init__.py         <- Makes pool a Python module that can be imported from everywhere.
│       └── ball_detection.py   <- Computer vision algorithm for detecting pool balls.
│       └── brain.py            <- Analytic solution of 4 types of pool shots.
│       └── calibration.py      <- camera calibration and inverse kinematics of the robot.
│       └── cam.py              <- Controls a DSLR camera using digiCamControl's command line interface.
│       └── dynamixel.py        <- Interacts with the Dynamixel servomotors.
│       └── error_analysis.py   <- Finds the error due to the vision and actuation system
│       └── eye.py              <- Corrects camera lens distortion and adjusts the perspective. 
│       └── pool_env.py         <- Defines a custom environment for simulating a pool game.
│       └── pool_frame.py       <- Creates pool table dimansions and geometry
│       └── pool_sim.py         <- Simulates the physics of a pool game using the Pymunk library.
│       └── random_balls.py     <- Generates random pool ball positions.
│       └── shot_selection.py   <- Chooses what pool shot to play based on the ball positions.
│       └── stepper.py          <- Interacts with the Arduino to send commands to the stepper motors.
│       └── utils.py            <- Contains several system parameters and basic functions.
│
└── LICENSE.txt

```

## Geometry of a pool shot

For the target ball `T` to pass through point `P`, the center of the pocket, the cue ball `C` must strike point `X`. 

Key measures:
- Cut angle: &gamma;
- Cue ball angle: &theta;
- Cue ball impact point: X 


<img src="./figures/geometry_of_the_pool_shot.png" alt="drawing" width="50%"/>

## Vision system

### Hardware

The hardware includes both the camera and its mounting support. The final setup is shown here:


| Top view | Side view |
|:------------:|:---------:|
| <img src="./figures/vision_system_1.png" alt="Top view" width="80%"/> | <img src="./figures/vision_system_2.png" alt="Side view" width="80%"/> |


To select the right camera, I conducted a resolution test on four different cameras I had at home: a smartphone camera, a Raspberry Pi camera, a DSLR, and a webcam. Based on the results and ease of connectivity with a PC, the final selection was the DSLR camera:


<table>
  <tr>
    <th>Camera</th>
    <td>Samsing A71</td>
    <td>PiCamera</td>
    <td>Canon EOS 1300D</td>
    <td>Logitech c922</td>
  </tr>

  <tr>
    <th>Resolution</th>
    <td>0.28 mm/px</td>
    <td>0.50 mm/px</td>
    <td>0.25 mm/px</td>
    <td>0.50 mm/px</td>
  </tr>

  <tr>
    <th>USAF test</th>
    <td><img src="./figures/mm_per_pixel_1.png" alt="drawing" width="90%"/>
</td>
    <td><img src="./figures/mm_per_pixel_2.png" alt="drawing" width="90%"/>
</td>
    <td><img src="./figures/mm_per_pixel_3.png" alt="drawing" width="100%"/>
</td>
    <td><img src="./figures/mm_per_pixel_4.png" alt="drawing" width="90%"/>
</td>
  </tr>

</table>



### Software

#### Preprocessing

- **Lens correction**: prevents straight lines from appearing curved.


| **Before**   | <img src="./figures/lens_correction_1.png" alt="drawing" width="100%"/>       |
|----------------|-----------------|
| **After**   | <img src="./figures/lens_correction_2.png" alt="drawing" width="100%"/>     |


- **Homography**: Transforms the image so that its corners align with those of the pool table.


| Before | After |
|----------|--------------|
| <img src="./figures/homography_1.png" alt="Before" width="100%"/> | <img src="./figures/homography_2.png" alt="After" width="100%"/> |

- **Camera calibration**: Computes the intrinsic camera parameters using images of a checkerborad pattern.


| Image 1 | Image 2 | Image 3 | Image 4 |
|:-----:|:-----:|:-----------------:|:-----------------:|
| <img src="./figures/camera_calibration_1.png" alt="2 DOF" width="100%"/> | <img src="./figures/camera_calibration_2.png" alt="2 DOF" width="100%"/> | <img src="./figures/camera_calibration_3.png" alt="2 DOF" width="100%"/> | [...]


#### YOLOv8

This approach uses the YOLO (version 8 by Ultralytics) detection algorithm, where the image is processed through a CNN, resulting in predictions in the form of rectangles.

<img src="./figures/yolo_algorithm.png" alt="drawing" width="100%"/>

#### Classic approach

This method employs a set of classic computer vision techniques: color segmentation, morphological operations, the watershed algorithm and color transformation. 

<img src="./figures/classic_algorithm.png" alt="drawing" width="100%"/>



#### Comparison of the results

After comparing the results of both methods against a Ground Truth established by me, it was observed that the classic method is more accurate, but less robust in situations where the balls are very close to each other. 


| Original | Ground truth | Classic | YOLOv8 |
|----------|--------------|---------|--------|
| <img src="./figures/results_1.png" alt="Original" width="100%"/> | <img src="./figures/results_2.png" alt="Ground truth" width="100%"/> | <img src="./figures/results_3.png" alt="Classic" width="100%"/> | <img src="./figures/results_4.png" alt="YOLOv8" width="100%"/> |



| Algorithm | &mu; [px] | &sigma; [px] |
|----------|----------|----------|
| YOLO    | 3.02     | 3.17     |
| Classic    | 2.43     | 1.81     |

If the 68–95–99.7 rule is applied, from these results it can be deduced that in 95% of the cases an error below 6.3 and 3.6 pixels should be expected for the YOLO and classic algorithms, respectively. Considering a resolution of 0.25 mm/px, this is 1.5 and 0.9 mm, respectively.

## Shot selection 

### Analytic method

- **Pool table geometry simplification**: to simplify the computation of rebounds.

| Original | Simplified |
|----------|--------------|
| <img src="./figures/pool_table_simplification_1.png" alt="Original" width="100%"/> | <img src="./figures/pool_table_simplification_2.png" alt="Ground truth" width="100%"/> |

- **Compute all possible and impossible shots**: combinations of all the different balls and pockets.


| CTP | CTTP | CBTP | CTBP |
|----------|--------------|---------|--------|
| <img src="./figures/CTP.png" alt="CTP" width="100%"/> | <img src="./figures/CTTP.png" alt="CTTP" width="100%"/> | <img src="./figures/CBTP.png" alt="CBTP" width="100%"/> | <img src="./figures/CTBP.png" alt="CTBP" width="100%"/> |

- **Filter feasible shots**: collision check using the geometric definition of the cross and dot products.


| No collision | Collision |
|:------------:|:---------:|
| <img src="./figures/collision_check_1.png" alt="No collision" width="50%"/> | <img src="./figures/collision_check_2.png" alt="Collision" width="50%"/> |


- **Choosing the easiest shot**: difficulty metric (using the nomenclature defined above)

$$
\text{difficulty} = \frac{ab}{\cos^2 \gamma}
$$

### Simulation method

- **2D simulation & render**: Pymunk, Pygame
- **Brute force**: every angle is simulated (incr. 0.1º)
- **Easiest shot**: less sensitive angle

<img src="./figures/simulation.gif" alt="drawing" width="60%"/>


## Actuation system: 

The pool robot is divided in 2 subsystems:

- **Cue ball impact**: linear actuator
   - Solenoid
   - Servomotor
- **Cue positioning**: cartesian robot
   - Stepper motors
   - V-slot 2020 and wheels
   - Tooth belt and timing pulleys 

Block diagram of the actuation system showing selected components and
interactions:

<img src="./figures/robot_solution.png" alt="drawing" width="90%"/>

- **Homing position**: it's necessary to have a reference position known as home since the stepper motors do not have a way to know their position. This position has been obtained using two limit switches activated by the robot's own moving parts:

<table>
  <tr>
    <th>SolidWorks</th>
    <th>Constructed</th>
  </tr>
  <tr>
    <td rowspan="2"><img src="./figures/homing_position_1.png" alt="SolidWorks" width="100%"/>
</td>
    <td><img src="./figures/homing_position_2.png" alt="Constructed" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="./figures/homing_position_3.png" alt="Constructed" width="100%"/></td>
  </tr>
</table>


- **Timing belt and pulleys**: the belt has been designed with an H configuration which has allowed the motors to be stationary

<table>
  <tr>
    <th>Concept design H-bot</th>
    <th>Constructed</th>
  </tr>
  <tr>
    <td rowspan="2"><img src="./figures/timing_belt_and_pulleys_1.png" alt="SolidWorks" width="100%"/>
</td>
    <td><img src="./figures/timing_belt_and_pulleys_2.png" alt="Constructed" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="./figures/timing_belt_and_pulleys_3.png" alt="Constructed" width="100%"/></td>
  </tr>
</table>

- **Belt tensioning and stepper sizing**: a set of pulleys was assembled and the belt was tensioned with zip ties. To achieve this the motors were sized with a calculation of the required torque, considering the accelaration and load terms, and a safety factor

| Belt tensioning | Stepper mount |
|----------|--------------|
| <img src="./figures/belt_tensioning.png" alt="Before" width="100%"/> | <img src="./figures/stepper_sizing.png" alt="After" width="73%"/> |

### Fliper design:

- **Why a flipper mechanism was developed?**: The cue ball impact subsystem was designed to be a flipper mechanism becasue, otherwise, the vertical postion of linear actuator had to be controlled:

| 2 DOF | 1 DOF |
|:------------:|:---------:|
| <img src="./figures/flipper_design_1.png" alt="2 DOF" width="60%"/> | <img src="./figures/flipper_design_2.png" alt="1 DOF" width="60%"/> |

- **Iterative design process**: The first iteration was entirely made with 3D-printed parts and served to test the viability of the concept. This proved successful, leading to a second iteration where the piece was modified to integrate a component from a pinball machine, in order to make it more robust. The final step involved rotating the flipper relative to the Cartesian robot's carriage. To accomplish this, a servo motor was coupled between the two parts.

| Proof of concept | Pinball parts | Servo integration |
|:-----:|:-----:|:-----------------:|
| <img src="./figures/iterative_design_1.png" alt="2 DOF" width="90%"/> | <img src="./figures/iterative_design_2.png" alt="2 DOF" width="90%"/> | <img src="./figures/iterative_design_3.png" alt="2 DOF" width="90%"/> |


- **Solenoid stroke and spring**:  the solenoid's stroke was limited using a screw, and the flipper's return to the retracted position was achieved with a spring.

Stroke                       |            Spring
:-------------------------:|:-------------------------:
<img src="./figures/solenoid_stroke.png" alt="drawing" width="100%"/>|  <img src="./figures/spring.png" alt="drawing" width="100%"/>


### Flipper parts:

- **Linkage kinematics simulation**: to obtain the dimensions of the mechanism, its kinematics have been simulated for many possible configurations, and the one that met a series of requirements was selected. For example, ensuring that the flipper, when retracted, does not collide with any balls that might be on the table.

- **Flipper assembly**: to secure the wooden flipper to the steel shaft, Araldite was used, while the shaft was joined to the metal part through welding. 

Flipper assembly           |            Flipper parts
:-------------------------:|:-------------------------:
<img src="./figures/flipper_1.png" alt="drawing" width="92%"/>|  <img src="./figures/flipper_2.png" alt="drawing" width="100%"/>


### Control and communications: 

- **Stepper motor**:
   - Accelstepper library
   - Pulse control: steps and direction
   - Stepper driver: DRV8825
   
- **Servomotor**:
   - Dynamixel library: address writing 
   - Trapezoidal velocity profile
   - PID angular position

- **Solenoid**:
   - Logic MOSFET circuit

- **PC interface**:
   - Arduino: serial communication 
   - USB communication converter 


### Final prototype:

The cartesian robot is supported by a wooden frame, and the flipper is held in place thanks to the sliding wheel element of the central bar, which is inverted.

SolidWorks                      |            Constructed
:-------------------------:|:-------------------------:
<img src="./figures/final_assembly_1.png" alt="drawing" width="100%"/>|  <img src="./figures/final_assembly_2.png" alt="drawing" width="100%"/>

## System integration

Find relationship between:

- Outputs from the vision system: cue ball pixel coordinates `uc`, `vc`
- Outputs from the shot selection algorithm: cue ball trajectory angle &theta;
- Commands to be sent to the robot's actuators: `q1`, `q2`, `q3`


Initially, it was necessary to study the kinematics of the Cartesian robot, observing the relationship between the motors' rotational movement and the linear displacement of the carriage. Then, by using a piece of cardboard and a marker, the relationship between the motor steps and the centimeters moved was determined.

Secondly, the movement required to position the flipper beneath the white ball was analyzed, taking into account the offset with the axis of rotation. It was deduced that the white ball's position in pixels, and the cosine and sine of the flipper's angle, have a linear relationship with the steps of the motors. However, the angle of the flipper matches the position to which we need to move the servo motor.


## Results

TODO



<details><summary><b>Example of use</b></summary>

1. Import libraries and data:

    ```python
   import numpy as np
   import tensorflow as tf

   test = 0
    ```

</details>


[a relative link](examples/actuators/dynamixel_and_stepper_control.py)
