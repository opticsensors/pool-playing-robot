import numpy as np
import pandas as pd
import cv2
from pool.brain import Brain

d_centroids={2:(1204,1268),
             8:(572,652),
             0:(2264,1476),
             7:(4132, 616),
             3: (516,2172)
             }

brain=Brain(d_centroids=d_centroids)
img = cv2.imread(f'./results/warp_0.jpg')
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
brain.setup_pockets(precision=6)
img=brain.draw_pool_balls(img)

# get balls that we want to pocket (coordinates and number id)
arr_balls_to_be_pocket = brain.get_balls_to_be_pocket(ball_type='solid')
#get special balls: cue ball and 8 ball
arr_cue, arr_8ball = brain.get_cue_and_8ball()
# get balls other balls for every cue and target (== for every target) that can intefere with the shot
other_balls = brain.get_other_balls(arr_balls_to_be_pocket)

#rename for clarity, using familiar nomenclature
C = arr_cue
T = other_balls
P = brain.pockets

#build array of combinations:
#combinations of C, T and P (T_ids is included because we are going to need ball info later)
C_comb,T_comb,P_comb = brain.get_point_combinations(C,T,P)

#get X1 and X2 (necessary point to check in what pockets ball can be pocketed)
T_comb_target_coord=T_comb[:,1:3]
X1_comb,X2_comb = brain.find_X1_and_X2(C_comb,T_comb_target_coord)

#create dataframe
df=pd.DataFrame({'Cx':C_comb[:,0],
                 'Cy':C_comb[:,1],
                 'T_id':T_comb[:,0],
                 'Tx':T_comb[:,1],
                 'Ty':T_comb[:,2],
                 'other_ball_id':T_comb[:,3],
                 'other_ball_x':T_comb[:,4],
                 'other_ball_y':T_comb[:,5],
                 'P_id':P_comb[:,0],
                 'Px':P_comb[:,1],
                 'Py':P_comb[:,2],
                 'X1x':X1_comb[:,0],
                 'X1y':X1_comb[:,1],
                 'X2x':X2_comb[:,0],
                 'X2y':X2_comb[:,1],
                 'Cx':C_comb[:,0],
                 'Cy':C_comb[:,1],
                 'Cx':C_comb[:,0],
                 'Cy':C_comb[:,1],})

#use above calculations to decide if that combination (row) is valid or not
valid_pockets = brain.find_valid_pockets(df[['Tx', 'Ty']].values,
                                         df[['Px', 'Py']].values,
                                         df[['X1x', 'X1y']].values,
                                         df[['X2x', 'X2y']].values)

#update variables
#C_valid = C_comb[valid_pockets]
#T_valid = T_comb[valid_pockets]
#P_valid = P_comb[valid_pockets]
df_valid=df[valid_pockets].copy()

#find_geometric_parameters
d,b,a,alpha, beta, X_comb = brain.find_geometric_parameters(df_valid[['Cx', 'Cy']].values,
                                                       df_valid[['Tx', 'Ty']].values,
                                                       df_valid[['Px', 'Py']].values)
df_valid['d']=d
df_valid['b']=b
df_valid['a']=a                                                               
df_valid['alpha']=alpha
df_valid['beta']=beta
df_valid['Xx']=X_comb[:,0]
df_valid['Xy']=X_comb[:,1]

df_valid.to_csv(path_or_buf='./data/ball_trajectories.csv', sep=',',index=False)

# collisions between C and other balls in CX trajectory
collision_configs_CX=brain.find_valid_trajectories(origin=df_valid[['Cx', 'Cy']].values,
                                            destiny=df_valid[['Xx', 'Xy']].values,
                                            collision_balls=df_valid[['other_ball_x', 'other_ball_y']].values)

# collisions between T and other balls in TP trajectory
collision_configs_TP=brain.find_valid_trajectories(origin=df_valid[['Tx', 'Ty']].values,
                                            destiny=df_valid[['Px', 'Py']].values,
                                            collision_balls=df_valid[['other_ball_x', 'other_ball_y']].values)

collision_configs=((collision_configs_CX) | (collision_configs_TP))
df_collisions=df_valid[collision_configs]
invalid_T=np.unique(df_collisions['T_id'])
invalid_P=np.unique(df_collisions['P_id'])

df_without_collisions=df_valid[~(df_valid['T_id'].isin(invalid_T) & df_valid['P_id'].isin(invalid_P))]


#filter by difficulty metric


#for point1, point2 in zip(df_valid[['X1x', 'X1y']].values, 
#                          df_valid[['Tx', 'Ty']].values): 
#    cv2.line(img, point1.astype(int), point2.astype(int), [26, 163, 251], 1)
#
#for point1, point2 in zip(df_valid[['X2x', 'X2y']].values, 
#                          df_valid[['Tx', 'Ty']].values): 
#    cv2.line(img, point1.astype(int), point2.astype(int), [26, 163, 251], 1)

for point1, point2 in zip(df_without_collisions[['Cx', 'Cy']].values, 
                          df_without_collisions[['Xx', 'Xy']].values): 
    cv2.line(img, point1.astype(int), point2.astype(int), [251, 163, 26], 1) 

for point1, point2 in zip(df_without_collisions[['Tx', 'Ty']].values, 
                          df_without_collisions[['Px', 'Py']].values): 
    cv2.line(img, point1.astype(int), point2.astype(int), [251, 163, 26], 1) 

cv2.imwrite('./results/pool_frame.png', img)