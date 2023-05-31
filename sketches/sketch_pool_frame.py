import numpy as np
import cv2
from pool.brain import Brain

d_centroids={2:(1204,1268),
             13:(572,652),
             0:(2264,1476),
             7:(4132, 616),
             8: (4000,2500)}

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
C_comb,T_comb,T_ids_comb,P_comb = brain.get_point_combinations(C,T,P)

#get X1 and X2 (necessary point to check in what pockets ball can be pocketed)
X1_comb,X2_comb = brain.find_X1_and_X2(C_comb,T_comb)

#use above calculations to decide if that combination (row) is valid or not
#valid_pockets = brain.find_valid_pockets(T_comb, P_comb, X1_comb, X2_comb)

#update variables
#C_valid = C_comb[valid_pockets]
#T_valid = T_comb[valid_pockets]
#T_ids_valid = T_ids_comb[valid_pockets]
#P_valid = P_comb[valid_pockets]

#find_geometric_parameters
#d,b,a,alpha, beta, X_valid = brain.find_geometric_parameters(C_valid,T_valid,P_valid)


# validate collisions between C and other balls in CX trajectory
# validate collisions between T and other balls in TP trajectory




#filter by difficulty metric

#for point1, point2 in zip(C_valid, X_valid): 
#    cv2.line(img, point1.astype(int), point2.astype(int), [251, 163, 26], 1) 
#
#for point1, point2 in zip(T_valid, P_valid): 
#    cv2.line(img, point1.astype(int), point2.astype(int), [251, 163, 26], 1) 
#
#cv2.imwrite('./results/pool_frame.png', img)
