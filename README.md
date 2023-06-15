# A pool playing robot project

A project that ...

# Installation

This command will clone the repository from GitHub:

```git clone https://github.com/opticsensors/pool-playing-robot.git```

Some of the commands require a newer version of pip, so start by making sure you have the latest version installed:

```py -m pip install --upgrade pip```

Make sure you have the latest version of PyPA’s build installed:

```py -m pip install --upgrade build```

Now run these commands from the same directory where pyproject.toml is located:

- ```py -m build```
- ```py -m pip install . ```   

# Dependencies
- opencv-python
- matplotlib
- scipy
- scikit-image
- opencv-contrib-python
- ultralytics
- pyserial

They should be installed automatically, but we can install them manually runing this command:

```py -m pip install <package_name>```

Note that stable-baselines3 was installed with:

```py -m pip install git+https://github.com/DLR-RM/stable-baselines3 ```

# Usage example

```python

from pool.eye import Eye
import cv2

eye=Eye()
#read image with random ball config 
img = cv2.imread('./data/config_3.jpg')
#get pool corners: top-right, top-left, bottom-left, bottom-right
corners = eye.get_pool_corners(img)

```