import numpy as np
import pandas as pd
import cv2
from pool.brain import Brain,PoolFrame

d_centroids={2:(1200,1268),
             #7:(572,652),
             0:(2264,1476),
             8:(4132, 616),
             3: (516,2172),
             14: (4132,2172),
             6: (1210,1800)
             }
img = cv2.imread(f'./results/warp_corners.jpg')

pool_frame=PoolFrame(img, 
                    precision=0,
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
img = pool_frame.draw_frame(img)

brain=Brain(pool_frame,
            d_centroids)

img=brain.draw_pool_balls(img)

df=brain.CBTP_shots('solid')

img=brain.draw_trajectories(img,df[['Bx', 'By']].values, 
                            df[['Cx', 'Cy']].values)

img=brain.draw_trajectories(img,df[['Bx', 'By']].values, 
                            df[['Xx', 'Xy']].values)

img=brain.draw_trajectories(img,df[['Tx', 'Ty']].values, 
                          df[['Px', 'Py']].values) 

cv2.imwrite('./results/pool_frame.png', img)