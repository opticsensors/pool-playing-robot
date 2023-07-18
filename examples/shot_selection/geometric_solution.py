import time
import cv2
from pool.random_balls import RandomBalls
from pool.shot_selection import GeomericSolution

img = cv2.imread(f'./results/warp_corners.jpg')

random_balls=RandomBalls()
d_centroids=random_balls.generate_random_balls()
print(d_centroids)

start = time.time()
gs = GeomericSolution()
angle = gs.get_actuator_angle(d_centroids, 'solid')
print(angle)
end = time.time()
print('time: ', end - start)

img_ctp, df = gs.debug(img, d_centroids, 'solid', 'CTP')
img_cbtp, df = gs.debug(img, d_centroids, 'solid', 'CBTP')
img_cttp, df = gs.debug(img, d_centroids, 'solid', 'CTTP')
img_ctbp, df = gs.debug(img, d_centroids, 'solid', 'CTBP')

cv2.imwrite('./results/geomeric_solution_ctp.png', img_ctp) 
cv2.imwrite('./results/geomeric_solution_cbtp.png', img_cbtp) 
cv2.imwrite('./results/geomeric_solution_cttp.png', img_cttp) 
cv2.imwrite('./results/geomeric_solution_ctbp.png', img_ctbp) 

