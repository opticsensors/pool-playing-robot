import cv2
from pool.random_balls import RandomBalls
from pool.shot_selection import GeomericSolution, BruteForce

img = cv2.imread(f'./results/warp_corners.jpg')

random_balls=RandomBalls()
d_centroids=random_balls.generate_random_balls()
print(d_centroids)

print('start Geomeric Solution')
gs = GeomericSolution()
angle = gs.get_actuator_angle(d_centroids, 'solid')
print(angle)

img, df = gs.debug(img, d_centroids, 'solid', 'CTBP')
cv2.imwrite('./results/geomeric_solution.png', img) # TODO change path
print('end Geomeric Solution')

print('start Brute Force')
bf = BruteForce()
angle = bf.get_actuator_angle(d_centroids, 'solid')
print(angle)

bf.debug(d_centroids, 'solid', angle)
print('end Brute Force')
