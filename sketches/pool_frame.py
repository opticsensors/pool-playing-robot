import cv2
import numpy as np
from pool.pool_frame import PoolFrame,Pockets, Cushions
from pool.random_balls import RandomBalls
from pool.brain import CTP, CBTP, CTTP, CTBP
from pool.utils import Rectangle, Params

params=Params()
img = cv2.imread(f'./results/warp_corners.jpg')
H=img.shape[0]
W=img.shape[1]

pockets=Pockets(pockets = params.POCKETS,
                computation_rectangle = params.COMPUTATIONAL_RECTANGLE,
                precision=0)

cushions=Cushions(computation_rectangle = params.COMPUTATIONAL_RECTANGLE,
                  cushion_ranges=params.CUSHION_RANGES)

pool_frame=PoolFrame(pockets,cushions)
img=pool_frame.draw_frame(img)
cv2.imwrite('./results/pool_frame.png', img)

random_balls=RandomBalls(ball_radius=102,
                         computation_rectangle = params.COMPUTATIONAL_RECTANGLE)
d_centroids=random_balls.generate_random_balls()
print(d_centroids)

ctp=CTP(pool_frame)
cbtp=CBTP(pool_frame)
cttp=CTTP(pool_frame)
ctbp=CTBP(pool_frame)

df_ctp=ctp.selected_shots(d_centroids, 'solid')
df_cbtp=cbtp.selected_shots(d_centroids, 'solid')
df_cttp=cttp.selected_shots(d_centroids, 'solid')
df_ctbp=ctbp.selected_shots(d_centroids, 'solid')

print(df_ctp)

img=ctp.draw_pool_balls(d_centroids,img)
img=ctp.draw_all_trajectories(df_ctp, img)
cv2.imwrite('./results/selected_shots.png', img)


