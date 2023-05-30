import numpy as np
import cv2
from pool.brain import Brain

brain=Brain()
img = cv2.imread(f'./results/warp_0.jpg')
img = brain.pool_frame(img, 
                    x_pocket_top_left=120,
                    y_pocket_top_left=96,
                    x_pocket_top_middle=2430,
                    y_pocket_top_middle=164,
                    x_pocket_bottom_middle=2427,
                    y_pocket_bottom_middle=2574,
                    horizontal_left_offset=250,
                    vertical_top_offset=260,
                    vertical_bottom_offset=240,
                    width_corner=80,
                    width_middle=70)

cv2.imwrite('./results/pool_frame.png', img)
