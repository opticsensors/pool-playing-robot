from pool import error_analysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#pool table
img = mpimg.imread('./data/pool_table.png')

#radii pool balls
r=38/2

#pool table dimensions
H=524
W=924
num_real_points=8
num_estimated_points=5
safety_distance=3*r #so the balls dont touch the pool table walls
maximum_error_vision_system=1 #units in mm

#generate real points for C and T
real_points=error_analysis.generate_random_numbers_inside_rectangle(W,H,num_real_points,safety_distance)

#pockets are fixed in space and we know their position
pockets=np.array([[38.447187,485.60708],
                  [38.447187,38.39291465],
                  [462.2112483,38.39291465],
                  [885.9753086,38.39291465],
                  [885.9753086,485.60708],
                  [462.2112483,485.60708]])

# generate combinations of all pairs of points in real_points
indices=np.transpose(np.triu_indices(real_points.shape[0],1))

#generate the pair permutations of points in real_points
#import itertools
#indices=np.array(list(itertools.permutations(list(range(num_real_points)),2)))

for idxC, idxT in indices:
    C=real_points[idxC,:]
    T=real_points[idxT,:]
    d=np.sqrt((C[0]-T[0])**2+(C[1]-T[1])**2)

    # we need to check if the configuration is possible (dist(C,T)>2*r)
    # we take 2.5 to make things clearer
    if d> 2.5*r:

        fig, ax = plt.subplots(num=f"{idxC}-{idxT}")
        ax.set_xlim((0, W))
        ax.set_ylim((0, H))
        ax=error_analysis.draw_pool_table_with_pockets(ax,W,H,img,pockets)

        slope1,slope2,intercept1,intercept2,reachable_pockets=error_analysis.pockets_inside_region_of_interest(pockets,C,T,d,2*r)
        ax=error_analysis.draw_region_of_interest(ax,W,C,T,slope1,slope2,intercept1,intercept2)
        
        print('num reachable pockets:',reachable_pockets.shape[0])
        print('combination:',f"{idxC}-{idxT}")

        for i,P in enumerate(reachable_pockets):
            
            print(f'==========================pocket_{i}=========================')
            _,_,_,_,_, X = error_analysis.geometric_parameters(r,C,T,P)
            print('C: ',C, 'T: ',T,'P: ',P, 'X', X)
            ax=error_analysis.draw_ideal_configuration(ax,r,C,T,P,X)
            
            #measured points
            C_estimated=error_analysis.generate_random_numbers_inside_circle(C,maximum_error_vision_system,num_estimated_points)
            T_estimated=error_analysis.generate_random_numbers_inside_circle(T,maximum_error_vision_system,num_estimated_points)
            P_estimated=error_analysis.generate_random_numbers_inside_circle(P,0.1,num_estimated_points)

            #since functions are not vectorized:
            for c_estimated,t_estimated,p_estimated in zip(C_estimated,T_estimated,P_estimated):
                _,_,_,_,_, X_estimated = error_analysis.geometric_parameters(r,c_estimated,t_estimated,p_estimated)
                X_calculated,Q,delta=error_analysis.compute_Q(r,C,T,P,c_estimated,X_estimated)
                
                print('Xcalculated',X_calculated)
                ax=error_analysis.draw_point(ax,c_estimated)
                ax=error_analysis.draw_point(ax,t_estimated)
                ax=error_analysis.draw_real_configuration(ax,r,C,X_calculated,Q)
    print('---------------------------------------------------------------------')

    ax.plot()  
    plt.show()


