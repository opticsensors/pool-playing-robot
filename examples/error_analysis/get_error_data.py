from pool import error_analysis
import numpy as np
import itertools
import pandas as pd


list_of_dict=[]
dict_to_save={}

#radii pool balls
r=38/2

#pool table dimensions
H=524
W=924
num_real_points=130
num_estimated_points=15
safety_distance=3*r #so the balls dont touch the pool table walls
maximum_error_vision_system=0.5 #units in mm

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
#indices=np.array(list(itertools.permutations(list(range(num_real_points)),2)))

for real_point_id, (idxC, idxT) in enumerate(indices):
    C=real_points[idxC,:]
    T=real_points[idxT,:]
    d=np.sqrt((C[0]-T[0])**2+(C[1]-T[1])**2)
    
    # we need to check if the configuration is possible (dist(C,T)>2*r)
    # we take 2.5 to make things clearer
    if d> 2.5*r:

        slope1,slope2,intercept1,intercept2,reachable_pockets=error_analysis.pockets_inside_region_of_interest(pockets,C,T,d,2*r)

        for reachable_pocket_id,P in enumerate(reachable_pockets):
            
            d,b,a,alpha, beta, X = error_analysis.geometric_parameters(r,C,T,P)
    
            #measured points
            C_estimated=error_analysis.generate_random_numbers_inside_circle(C,maximum_error_vision_system,num_estimated_points)
            T_estimated=error_analysis.generate_random_numbers_inside_circle(T,maximum_error_vision_system,num_estimated_points)
            P_estimated=error_analysis.generate_random_numbers_inside_circle(P,0.1,num_estimated_points)

            #since functions are not vectorized:
            for estimated_point_id, (c_estimated,t_estimated,p_estimated) in enumerate(zip(C_estimated,T_estimated,P_estimated)):
                _,_,_,_,_, X_estimated = error_analysis.geometric_parameters(r,c_estimated,t_estimated,p_estimated)
                X_calculated,Q,delta=error_analysis.compute_Q(r,C,T,P,c_estimated,X_estimated)
                
                dict_to_save['real_point_id']=real_point_id
                dict_to_save['reachable_pocket_id']=reachable_pocket_id
                dict_to_save['estimated_point_id']=estimated_point_id
                dict_to_save['C_x']=C[0]
                dict_to_save['C_y']=C[1]
                dict_to_save['T_x']=T[0]
                dict_to_save['T_y']=T[1]
                dict_to_save['P_x']=P[0]
                dict_to_save['P_y']=P[1]
                dict_to_save['X_x']=X[0]
                dict_to_save['X_y']=X[1]
                dict_to_save['C_estimated_x']=c_estimated[0]
                dict_to_save['C_estimated_y']=c_estimated[1]
                dict_to_save['T_estimated_x']=t_estimated[0]
                dict_to_save['T_estimated_y']=t_estimated[1]
                dict_to_save['P_estimated_x']=p_estimated[0]
                dict_to_save['P_estimated_y']=p_estimated[1]
                dict_to_save['X_estimated_x']=X_estimated[0]
                dict_to_save['X_estimated_x']=X_estimated[1]
                dict_to_save['X_calculated_x']=X_calculated[0]
                dict_to_save['X_calculated_y']=X_calculated[1]
                dict_to_save['Q_x']=Q[0]
                dict_to_save['Q_y']=Q[1]
                dict_to_save['d']=d
                dict_to_save['b']=b
                dict_to_save['a']=a
                dict_to_save['alpha']=alpha
                dict_to_save['beta']=beta
                dict_to_save['delta']=delta

                list_of_dict.append(dict_to_save.copy())

        
#for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf=f'./results/error_data_{maximum_error_vision_system}_mm.csv', sep=' ',index=False)


