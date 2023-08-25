import time
import cv2
from pool.random_balls import RandomBalls
from pool.shot_selection import BruteForce

img = cv2.imread(f'./results/warp_corners.jpg')

random_balls=RandomBalls()
d_centroids=random_balls.generate_random_balls()
#d_centroids={0: (2680, 1412), 8: (4083, 846), 3:(516,2172), 4:(1492,1436), 14:(4329,1900)}
print(d_centroids)

start = time.time()
bf = BruteForce()
angle = bf.get_actuator_angle(d_centroids, 'solid')
print(angle)
end = time.time()
print('time: ', end - start)

bf.debug(d_centroids, 'solid', angle, 'array')

