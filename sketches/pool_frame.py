import cv2
import numpy as np
from pool.pool_frame import PoolFrame,Pockets, Cushions
from pool.random_balls import RandomBalls
from pool.brain import CTP, CBTP, CTTP, CTBP
from pool.utils import Rectangle

img = cv2.imread(f'./results/warp_corners.jpg')
H=img.shape[0]
W=img.shape[1]

img_rectangle=Rectangle(top_left=(0,0),
                        bottom_right=(W,H))
computation_rectangle=img_rectangle.get_rectangle_with_offsets((260, 250, 240, 250))
#pockets_corners_rect=img_rectangle.get_rectangle_with_offsets((96, 120, 96, 120))

pockets = [(96, 120),      # top left          
           (96, H-120),    # top right       
           (W-96, H-120),  # bottom right     
           (96, H-120),    # bottom left
           (2430, 164),    # top middle    
           (2430, 2574)]   # bottom middle    


pockets=Pockets(pockets = pockets,
                computation_rectangle = computation_rectangle,
                precision=0)

cushions=Cushions(computation_rectangle = computation_rectangle,
                  cushion_ranges=((445,2154),(2729,4338),(476,2223)))

pool_frame=PoolFrame(pockets,cushions)
img=pool_frame.draw_frame(img)
cv2.imwrite('./results/pool_frame.png', img)

random_balls=RandomBalls(ball_radius=102,
                         computation_rectangle = computation_rectangle)
d_centroids=random_balls.generate_random_balls()

ctp=CTP(pool_frame, 102)
df_ctp=ctp.selected_shots(d_centroids, 'solid')
img=ctp.draw_pool_balls(d_centroids,img)
img=ctp.draw_all_trajectories(df_ctp, img)
cv2.imwrite('./results/selected_shots.png', img)


