import numpy as np
import pandas as pd
import cv2
from pool.brain import Brain

d_centroids={2:(1200,1268),
             7:(572,652),
             0:(2264,1476),
             8:(4132, 616),
             3: (516,2172),
             4: (2300,2100),
             6: (1210,1600)
             }

brain=Brain(d_centroids=d_centroids)
img = cv2.imread(f'./results/warp_corners.jpg')
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
brain.setup_pockets(precision=10)
img=brain.draw_pool_balls(img)

# get balls that we want to pocket (coordinates and number id)
arr_balls_to_be_pocket = brain.get_balls_to_be_pocket(ball_type='solid')
#get special balls: cue ball and 8 ball
arr_cue, arr_8ball = brain.get_cue_and_8ball()
# get other balls for every cue and target (== for every target) that can intefere with the shot
other_balls=brain.get_other_balls_twice(arr_balls_to_be_pocket, 'solid')

#rename for clarity, using familiar nomenclature
C = arr_cue
T = other_balls
P = brain.pockets

#build array of combinations:
comb=brain.get_row_combinations_of_two_arrays(T,P)
comb=brain.get_row_combinations_of_two_arrays(C,comb)

#create dataframe
df=pd.DataFrame({'Cx':comb[:,0],
                 'Cy':comb[:,1],
                 'T_id':comb[:,2],
                 'Tx':comb[:,3],
                 'Ty':comb[:,4],
                 'B_id':comb[:,5],
                 'Bx':comb[:,6],
                 'By':comb[:,7],
                 'other_ball_id':comb[:,8],
                 'other_ball_x':comb[:,9],
                 'other_ball_y':comb[:,10],
                 'P_id':comb[:,11],
                 'P_sub_id':comb[:,12],
                 'Px':comb[:,13],
                 'Py':comb[:,14]})

#get X1 and X2 (necessary point to check in what pockets ball can be pocketed)
X1_comb,X2_comb = brain.find_X1_and_X2(df[['Bx', 'By']].values,
                                       df[['Tx', 'Ty']].values)

df['X1x']=X1_comb[:,0]
df['X1y']=X1_comb[:,1]
df['X2x']=X2_comb[:,0]
df['X2y']=X2_comb[:,1]

#use above calculations to decide if that combination (row) is valid or not
valid_pockets = brain.find_valid_pockets(df[['Tx', 'Ty']].values,
                                         df[['Px', 'Py']].values,
                                         df[['X1x', 'X1y']].values,
                                         df[['X2x', 'X2y']].values)

df_valid=df[valid_pockets].copy()

#find_geometric_parameters
X_comb = brain.find_X(C=df_valid[['Bx', 'By']].values,
                                                       T=df_valid[['Tx', 'Ty']].values,
                                                       P=df_valid[['Px', 'Py']].values)
df_valid['Xx']=X_comb[:,0]
df_valid['Xy']=X_comb[:,1]

#df_valid.to_csv(path_or_buf='./data/ball_trajectories.csv', sep=',',index=False)

#find_geometric_parameters
X_new_comb = brain.find_X(C=df_valid[['Cx', 'Cy']].values,
                                                       T=df_valid[['Bx', 'By']].values,
                                                       P=df_valid[['Xx', 'Xy']].values)

df_valid['X_new_x']=X_new_comb[:,0]
df_valid['X_new_y']=X_new_comb[:,1]

#get X1 and X2 (necessary point to check in what pockets ball can be pocketed)
X1_comb,X2_comb = brain.find_X1_and_X2(df_valid[['Cx', 'Cy']].values,
                                       df_valid[['Bx', 'By']].values)

df_valid['X3x']=X1_comb[:,0]
df_valid['X3y']=X1_comb[:,1]
df_valid['X4x']=X2_comb[:,0]
df_valid['X4y']=X2_comb[:,1]

#use above calculations to decide if that combination (row) is valid or not
valid_pockets = brain.find_valid_pockets(df_valid[['Bx', 'By']].values,
                                         df_valid[['Xx', 'Xy']].values,
                                         df_valid[['X3x', 'X3y']].values,
                                         df_valid[['X4x', 'X4y']].values)

df_valid=df_valid[valid_pockets].copy()



# collisions between C and other balls in CX trajectory
collision_configs_CXnew=brain.find_collision_trajectories(origin=df_valid[['Cx', 'Cy']].values,
                                            destiny=df_valid[['X_new_x', 'X_new_y']].values,
                                            collision_balls=df_valid[['other_ball_x', 'other_ball_y']].values)

# collisions between T and other balls in TP trajectory
collision_configs_BX=brain.find_collision_trajectories(origin=df_valid[['Bx', 'By']].values,
                                            destiny=df_valid[['Xx', 'Xy']].values,
                                            collision_balls=df_valid[['other_ball_x', 'other_ball_y']].values)

# collisions between T and other balls in TP trajectory
collision_configs_TP=brain.find_collision_trajectories(origin=df_valid[['Tx', 'Ty']].values,
                                            destiny=df_valid[['Px', 'Py']].values,
                                            collision_balls=df_valid[['other_ball_x', 'other_ball_y']].values)


collision_configs=((collision_configs_CXnew) | (collision_configs_BX)| (collision_configs_TP))
df_collisions=df_valid[collision_configs]
arr_collisions=df_collisions[['T_id', 'B_id', 'P_id','P_sub_id']].values
arr_configs=df_valid[['T_id', 'B_id', 'P_id','P_sub_id']].values
collision_configs=(arr_configs[None,:]==arr_collisions[:,None]).all(-1).any(0)

df_without_collisions=df_valid[~(collision_configs)]


#filter by difficulty metric
df_filtered=df_without_collisions.copy()
df_filtered['XB_TX_abs_angle'] = brain.deviation_from_ideal_angle(df_filtered[['Tx', 'Ty']].values,
                                                                    df_filtered[['Xx', 'Xy']].values,
                                                                    df_filtered[['Bx', 'By']].values)
df_filtered=df_filtered[df_filtered['XB_TX_abs_angle'] < 45]

df_filtered['XC_BX_abs_angle'] = brain.deviation_from_ideal_angle(df_filtered[['Bx', 'By']].values,
                                                                    df_filtered[['X_new_x', 'X_new_y']].values,
                                                                    df_filtered[['Cx', 'Cy']].values)
df_filtered=df_filtered[df_filtered['XC_BX_abs_angle'] < 45]


img=brain.draw_trajectories(img,df_filtered[['Cx', 'Cy']].values, 
                          df_filtered[['X_new_x', 'X_new_y']].values)
img=brain.draw_trajectories(img,df_filtered[['Bx', 'By']].values, 
                          df_filtered[['Xx', 'Xy']].values)
img=brain.draw_trajectories(img,df_filtered[['Tx', 'Ty']].values, 
                          df_filtered[['Px', 'Py']].values) 

cv2.imwrite('./results/pool_frame.png', img)