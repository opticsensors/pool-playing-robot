# A pool playing robot project

Design and implementation of a robot that plays the game of pool autonomously. To achieve this goal, three different branches had to be explored:
- **Vision system**: computer vision algorithms to detect and identify the pool balls
- **Shot selection**: algorithms to find the *best* shot given the balls configuration and pool table dimensions
- **Actuators**: how to send commands to the actuators and the mechanical parts to make the objective task possible

# Installation

This command will clone the repository from GitHub:

```git clone https://github.com/opticsensors/pool-playing-robot.git```

Some of the commands require a newer version of pip, so start by making sure you have the latest version installed:

```py -m pip install --upgrade pip```

Make sure you have the latest version of PyPA’s build installed:

```py -m pip install --upgrade build```

Now run these commands from the same directory where pyproject.toml is located:

- ```py -m build```
- ```py -m pip install . ``` or ```py -m pip install -e . ``` when developing

# Dependencies
- keyboard
- dynamixel-sdk
- opencv-python
- matplotlib
- numpy
- scipy
- scikit-image
- opencv-contrib-python>=4.7
- ultralytics
- pyserial
- pandas
- scikit-image
- scikit-learn
- scipy
- gymnasium
- pygame
- pymunk

They should be installed automatically, but we can install them manually runing this command:

```py -m pip install <package_name>```

# Usage examples

Code examples of different robot subystems are shown in the ```examples/``` folder. 