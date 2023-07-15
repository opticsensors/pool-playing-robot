import cv2
import numpy as np
from pool.random_balls import RandomBalls
from pool.brain import ShotSelection

img = cv2.imread(f'./results/warp_corners.jpg')

#random_balls=RandomBalls()
#d_centroids=random_balls.generate_random_balls()
d_centroids={5.0: [1358.78857421875, 248.07107543945312], 0.0: [1528.2252197265625, 1716.3856201171875], 8.0: [909.4813842773438, 1999.3070068359375],}
print(d_centroids)

ss = ShotSelection()
angle = ss.get_actuator_angle(d_centroids, 'solid')
print(angle)

img, df = ss.debug(img, d_centroids, 'solid', 'CBTP')
print(df)

cv2.imwrite('./results/selected_shots.png', img) # TODO change path

