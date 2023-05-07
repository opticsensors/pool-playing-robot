import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# mechanism constants
l1_min=12
l1_max=17
l2_min=15
l2_max=20
x_min=7
x_max=33
a_min=45
a_max=60
b_min=20
b_max=30
d_min=18
d_max=25

a_arr=np.arange(a_min,a_max,1)
b_arr=np.arange(b_min,b_max,1)
d_arr=np.arange(d_min,d_max,1)
x_arr=np.arange(x_min,x_max,0.5)
l1_arr=np.arange(l1_min,l1_max,1)
l2_arr=np.arange(l2_min,l2_max,1)

prev_P=None
list_of_dict=[]
dict_to_save={}

combinations=np.array(np.meshgrid(-x_arr, d_arr, l1_arr,l2_arr, a_arr, b_arr)).T.reshape(-1,6)
A=combinations[:,:2]
l1_comb=combinations[:,2]
l2_comb=combinations[:,3]
a_comb=combinations[:,4]
b_comb=combinations[:,5]

# distance from P0 to P1    
d=np.sqrt((A[:,0])**2+(A[:,1])**2)
# distance from P0 to P2 (being P2 the point of intersection between lines P0P1 and X1X2)
# (X1,X2 are the intersection points that we want to find)
a=(d**2+l1_comb**2-l2_comb**2)/(2*d)
# distance from P1 to P2
b=d-a
# distance from P2 to X1 = distance from P2 to X2
h=np.sqrt(l1_comb**2-a**2)
# convert points to numpy array
C= np.zeros_like(A)
CA=C-A
auxiliar_points=(A.T * (b/d)).T+(C.T * (a/d)).T

intersec1_x=auxiliar_points[:,0]+(h/d)*CA[:,1]
intersec2_x=auxiliar_points[:,0]-(h/d)*CA[:,1]

intersec1_y=auxiliar_points[:,1]-(h/d)*CA[:,0]
intersec2_y=auxiliar_points[:,1]+(h/d)*CA[:,0]

#B=np.column_stack((intersec1_x,intersec1_y))
B=np.column_stack((intersec2_x,intersec2_y))

norm_BC_vector=l2_comb
BC_vector=(-B.T * (1/norm_BC_vector)).T
perpendicular_BC_vector=np.column_stack((BC_vector[:,1], -BC_vector[:,0]))    
P=(BC_vector.T * (a_comb)).T+(perpendicular_BC_vector.T * (b_comb)).T

combinations=np.concatenate((combinations,B, P), axis=1)

df=pd.DataFrame({'Ax': combinations[:, 0], # == -x
                 'Ay': combinations[:, 1], # == d
                 'l1': combinations[:, 2],  
                 'l2': combinations[:, 3],
                 'a':  combinations[:, 4],
                 'b':  combinations[:, 5],
                 'Bx': combinations[:, 6],
                 'By': combinations[:, 7],
                 'Px': combinations[:, 8],
                 'Py': combinations[:, 9],
                 })

df['config_id'] = df.groupby(['Ay', 'l1', 'l2', 'a', 'b',]).ngroup()
df.to_csv(path_or_buf='./data/all_mechanisms.csv', sep=' ',index=False)

list_of_dict=[]
dict_to_save={}

#find P(xmax)
for config in np.unique(df['config_id']):
    df_config=df[df['config_id']==config]
    idx=df_config.index[df_config['Px'].abs()==df_config['Px'].abs().min()]
    try:
        idx=idx[0]
    except IndexError:
        continue
    desired_x_max_row=df_config.iloc[[idx]]
    Px_xmax=desired_x_max_row['Px'].values
    Py_xmax=desired_x_max_row['Py'].values
    idx=df_config.index[(df_config['Py']-Py_xmax-20).abs()==(df_config['Py']-Py_xmax-20).abs().min()]
    try:
        idx=idx[0]
    except IndexError:
        continue
    desired_x_min_row=df_config.iloc[[idx]]
    Px_xmin=desired_x_min_row['Px'].values
    Py_xmin=desired_x_min_row['Py'].values

    dict_to_save['config_id']=config
    dict_to_save['xmin']=-desired_x_min_row['Ax'].values[0]
    dict_to_save['xmax']=-desired_x_max_row['Ax'].values[0]
    dict_to_save['d']=  df_config['Ay'][0]
    dict_to_save['l1']= df_config['l1'][0]
    dict_to_save['l2']= df_config['l2'][0]
    dict_to_save['a']=  df_config['a'][0]
    dict_to_save['b']=  df_config['b'][0]

    list_of_dict.append(dict_to_save.copy())

df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf=f'./data/valid_mechanisms.csv', sep=' ',index=False)