import numpy as np
import pandas as pd
from pool import utils

# mechanism constants
l1_min=20
l1_max=26
l2_min=23.6 # l2 is fixed by the metal piece
l2_max=24.6
x_min=10
x_max=40
a_min=40
a_max=60
b_min=0
b_max=1 # b is fixed by d and a
d_min=20
d_max=30

a_arr=np.arange(a_min,a_max,1)
b_arr=np.arange(b_min,b_max,1)
d_arr=np.arange(d_min,d_max,1)
x_arr=np.arange(x_min,x_max,0.5)
l1_arr=np.arange(l1_min,l1_max,1)
l2_arr=np.arange(l2_min,l2_max,1)

list_of_dict=[]
dict_to_save={}

combinations=np.array(np.meshgrid(-x_arr, d_arr, l1_arr,l2_arr, a_arr, b_arr)).T.reshape(-1,6)
A=combinations[:,:2]
C= np.zeros_like(A)
l1_comb=combinations[:,2]
l2_comb=combinations[:,3]
a_comb=combinations[:,4]
b_comb=combinations[:,5]

print('total rows',combinations.shape[0])

flipper_dist=63.7
def new_b(a,flipper_dist):
    return np.sqrt(flipper_dist**2-a**2)

b_comb=new_b(a_comb, flipper_dist)

_,B=utils.intersection_two_circles(A,C,l1_comb,l2_comb)

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

print('total configs',np.unique(df['config_id']).max())

angle=np.degrees(utils.angle_between_3_points(A,B,C))
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

print('total rows',df1.shape[0])

valid_configs=df1[df1['x_stroke']<=19]
valid_configs=valid_configs[valid_configs['angle']<140]
df1.to_csv(path_or_buf='./results/valid_mechanisms.csv', sep=',',index=False)

print(valid_configs)