import time
import cv2
from pool.random_balls import RandomBalls
from pool.shot_selection import GeomericSolution

img = cv2.imread(f'./results/warp_corners.jpg')

random_balls=RandomBalls()
d_centroids=random_balls.generate_random_balls()
#d_centroids={0: (2680, 1412), 8: (4083, 846), 3:(516,2172), 4:(1492,1436), 14:(4329,1900)}
print(d_centroids)

start = time.time()
gs = GeomericSolution()
angle = gs.get_actuator_angle(d_centroids, 'solid')
print(angle)
end = time.time()
print('time: ', end - start)

img_ctp,  df_ctp = gs.debug(img, d_centroids, 'solid', 'CTP')
img_cbtp, df_cbtp = gs.debug(img, d_centroids, 'solid', 'CBTP')
img_cttp, df_cttp = gs.debug(img, d_centroids, 'solid', 'CTTP')
img_ctbp, df_ctbp = gs.debug(img, d_centroids, 'solid', 'CTBP')

pool_frame=gs.pool_frame.draw_frame(img)
pool_frame_and_balls=gs.pool_frame.draw_pool_balls(pool_frame.copy(), d_centroids)
cv2.imwrite('./results/pool_frame.png', pool_frame)
cv2.imwrite('./results/pool_frame_and_balls.png', pool_frame_and_balls)
cv2.imwrite('./results/geomeric_solution_ctp.png', img_ctp) 
cv2.imwrite('./results/geomeric_solution_cbtp.png', img_cbtp) 
cv2.imwrite('./results/geomeric_solution_cttp.png', img_cttp) 
cv2.imwrite('./results/geomeric_solution_ctbp.png', img_ctbp) 

