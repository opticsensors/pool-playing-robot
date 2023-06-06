import numpy as np
import pandas as pd
import cv2
from pool.brain import Brain,PoolFrame

d_centroids={#2:(1200,1268),
             #7:(572,652),
             0:(2264,1476),
             8:(4132, 616),
             1:(2290,1850),
             #3: (516,2172),
             #14: (4132,2172),
             4: (2400,2200),
             #6: (1210,1800)
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
#cv2.imwrite('./results/pool_frame.png', img)

brain=Brain(pool_frame)

img=brain.draw_pool_balls(d_centroids, img)

df1=brain.CTP_shots(d_centroids,'solid')
df2=brain.CBTP_shots(d_centroids,'solid')
df3=brain.CTTP_shots(d_centroids,'solid')
df4=brain.CTBP_shots(d_centroids,'solid')

img=brain.wrapper_draw_trajectories(df4,img,'CTBP')

cv2.imwrite('./results/pool_frame.png', img)