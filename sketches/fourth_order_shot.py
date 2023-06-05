import numpy as np
import pandas as pd
import cv2
from pool.brain import Brain

d_centroids={2:(1200,1268),
             7:(572,652),
             0:(2264,1476),
             8:(4132, 616),
             #3: (516,2172),
             6: (1210,1800),
             #14: (4132,2172),
             }

brain=Brain(d_centroids=d_centroids)
img = cv2.imread(f'./results/warp_corners.jpg')
H=img.shape[0]
W=img.shape[1]
img = brain.setup_pool_frame(img, 
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
brain.setup_pockets(precision=0)
img=brain.draw_pool_balls(img)

# get balls that we want to pocket (coordinates and number id)
arr_balls_to_be_pocket = brain.get_balls_to_be_pocket(ball_type='solid')
#get special balls: cue ball and 8 ball
arr_cue, arr_8ball = brain.get_cue_and_8ball()
# get other balls for every cue and target (== for every target) that can intefere with the shot
other_balls = brain.get_other_balls(arr_balls_to_be_pocket)

#rename for clarity, using familiar nomenclature
C = arr_cue
T = other_balls
P = brain.pockets
T_reflect=brain.get_T_ball_reflections(T)

#build array of combinations:
comb=brain.get_row_combinations_of_two_arrays(T,P)
comb=brain.get_row_combinations_of_two_arrays(T_reflect,comb)
comb=brain.get_row_combinations_of_two_arrays(C,comb)

#create dataframe
df=pd.DataFrame({'Cx':comb[:,0],
                 'Cy':comb[:,1],
                 'T_reflect_id': comb[:,3],
                 'T_reflect_sub_id': comb[:,2],
                 'T_reflect_x':comb[:,4],
                 'T_reflect_y':comb[:,5],
                 'T_id':comb[:,6],
                 'Tx':comb[:,7],
                 'Ty':comb[:,8],
                 'other_ball_id':comb[:,9],
                 'other_ball_x':comb[:,10],
                 'other_ball_y':comb[:,11],
                 'P_id':comb[:,12],
                 'P_sub_id':comb[:,13],
                 'Px':comb[:,14],
                 'Py':comb[:,15]})
# only configs where the reflected ball id is the same as the T ball
df=df[df['T_reflect_id']==df['T_id']]

B_comb = brain.find_bouncing_points_v2(df[['T_reflect_x', 'T_reflect_y']].values,
                                    df[['Px', 'Py']].values,)
df['Bx']=B_comb[:,0]
df['By']=B_comb[:,1]

df[((df['P_id']==6) & (df['T_reflect_sub_id']==1)) |
   (df['P_id']==6 & df['T_reflect_sub_id']==2) |
   (df['P_id']==5 & df['T_reflect_sub_id']==1) |
   (df['P_id']==5 & df['T_reflect_sub_id']==2) |
   (df['P_id']==5 & df['T_reflect_sub_id']==4) |
   (df['P_id']==4 & df['T_reflect_sub_id']==1) |
   (df['P_id']==4 & df['T_reflect_sub_id']==4) |
   (df['P_id']==3 & df['T_reflect_sub_id']==3) |
   (df['P_id']==3 & df['T_reflect_sub_id']==4) |
   (df['P_id']==2 & df['T_reflect_sub_id']==2) |
   (df['P_id']==2 & df['T_reflect_sub_id']==3) |
   (df['P_id']==2 & df['T_reflect_sub_id']==4) |
   (df['P_id']==1 & df['T_reflect_sub_id']==3) |
   (df['P_id']==1 & df['T_reflect_sub_id']==2) ]


B_comb = brain.find_bouncing_points_v2(df[['T_reflect_x', 'T_reflect_y']].values,
                                    df[['Px', 'Py']].values,)
#find_geometric_parameters
X_comb = brain.find_X(  df[['Tx', 'Ty']].values,
                        df[['Bx', 'By']].values)
df['Xx']=X_comb[:,0]
df['Xy']=X_comb[:,1]


#get X1 and X2 (necessary point to check in what pockets ball can be pocketed)
X1_comb,X2_comb = brain.find_X1_and_X2(df[['Cx', 'Cy']].values,
                                       df[['Tx', 'Ty']].values)

df['X1x']=X1_comb[:,0]
df['X1y']=X1_comb[:,1]
df['X2x']=X2_comb[:,0]
df['X2y']=X2_comb[:,1]