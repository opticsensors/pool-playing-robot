# A pool playing robot project

A project that ...


# Dependencies
- python -m pip install opencv-python
- python -m pip install matplotlib
- python -m pip install scipy
- python -m pip install scikit-image
- python -m pip install opencv-contrib-python
- python -m pip install ultralytics

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