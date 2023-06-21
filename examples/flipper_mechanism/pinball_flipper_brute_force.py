import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# mechanism constants
l1_min=14
l1_max=30
l2_min=23.6
l2_max=24.6
x_min=-5
x_max=37
a_min=30
a_max=63
b_min=20
b_max=21
d_min=12
d_max=30

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

print('total rows',combinations.shape[0])

flipper_dist=63.7
def new_b(a,flipper_dist):
    return np.sqrt(flipper_dist**2-a**2)

new_b=new_b(a_comb, flipper_dist)
b_comb=new_b.copy()

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
                 'l1': l1_comb,  
                 'l2': l2_comb,
                 'a':  a_comb,
                 'b':  b_comb,
                 'Bx': combinations[:, 6],
                 'By': combinations[:, 7],
                 'Px': combinations[:, 8],
                 'Py': combinations[:, 9],
                 })

df['config_id'] = df.groupby(['Ay', 'l1', 'l2', 'a', 'b',]).ngroup()
df.to_csv(path_or_buf='./all_mechanisms.csv', sep=' ',index=False)

print('total configs',np.unique(df['config_id']).max())

def angle_abc(a,b,c):
    """
    Computes the angle between 3 points 
    (point b is the vertex)

    Parameters
    ----------    
        a,b,c: numpy array of shape (2,)
            x, y coordinates of the point
    Returns
    -------
        angle: numpy float64

    """
    ba = a - b
    bc = c - b
    dot = ba[:,0]*bc[:,0] + ba[:,1]*bc[:,1]
    cosine_angle = dot / (l1_comb * l2_comb)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

angle=angle_abc(A,B,C)
df['angle']=angle

list_of_dict=[]
dict_to_save={}
df=df.dropna()
for config in np.unique(df['config_id']):
    df_config=df[df['config_id']==config]
    df_xmax=df_config[df_config['Px'].abs()<2]
    
    if df_xmax.shape[0]!=0:
        desired_x_max_row=df_xmax[df_xmax['Px'].abs()==df_xmax['Px'].abs().min()]
        Py_xmax=desired_x_max_row['Py'].values
        df_xmin=df_config[(df_config['Py']-Py_xmax)>22]

        if df_xmin.shape[0]!=0:
            desired_x_min_row=df_xmin[(df_xmin['Py']-Py_xmax-22).abs()==(df_xmin['Py']-Py_xmax-22).abs().min()]
            Py_xmin=desired_x_min_row['Py'].values

            dict_to_save['config_id']=config
            dict_to_save['angle']=desired_x_max_row['angle'].values[0]
            dict_to_save['xmin']=-(desired_x_min_row['Ax'].values[0])
            dict_to_save['xmax']=-(desired_x_max_row['Ax'].values[0])
            dict_to_save['x_stroke']=abs(dict_to_save['xmax']-dict_to_save['xmin'])
            dict_to_save['d']=  df_config['Ay'].values[0]
            dict_to_save['l1']= df_config['l1'].values[0]
            dict_to_save['l2']= df_config['l2'].values[0]
            dict_to_save['a']=  df_config['a'].values[0]
            dict_to_save['b']=  df_config['b'].values[0]

            list_of_dict.append(dict_to_save.copy())
            print(config,df_xmax.shape, desired_x_max_row.shape, df_xmin.shape, desired_x_min_row.shape )

df1 = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df1.to_csv(path_or_buf='./valid_mechanisms.csv', sep=' ',index=False)

print('total rows',df1.shape[0])

valid_configs=df1[df1['x_stroke']<=19]
valid_configs=valid_configs[valid_configs['angle']<140]
valid_configs