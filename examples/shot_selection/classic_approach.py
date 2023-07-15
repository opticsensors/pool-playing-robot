import cv2
import numpy as np
from pool.random_balls import RandomBalls
from pool.brain import ShotSelection

img = cv2.imread(f'./results/warp_corners.jpg')

random_balls=RandomBalls()
d_centroids=random_balls.generate_random_balls()
d_centroids={0: (4306, 1998), 8: (4509, 609), 3: (1433, 2183), 5: (2326, 1085), 1: (3776, 494), 13: (2989, 2354)}
print(d_centroids)

ss = ShotSelection()
angle = ss.get_actuator_angle(d_centroids, 'solid')
print(angle)

img, df = ss.debug(img, d_centroids, 'solid', 'CTBP')
print(df)

cv2.imwrite('./results/selected_shots.png', img) # TODO change path

