import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model(time, tau, K):
    y = K*(1-np.exp(-time/tau))
    return y

l_Kp=[20,30,40,50,60]
for Kp in l_Kp:
    print(Kp)
    df = pd.read_csv(f'./step_data_p{Kp}.csv', sep=',',decimal='.')
    df['change']=df['Goal Position'].diff()
    print('num files to be saved: ', len(np.unique(df['change']))-2)
    df['error']=df['Goal Position']-df['Present Position']
    df['K*error']=Kp*df['error']
    df['Time[s]']=df['Time[ms]']/1000


    df=df.fillna(0)
    list_of_df=np.split(df,df[df['change']!=0].index)
    #first element of list should be removed because motor is stationary in the beginning
    list_of_df=list_of_df[1:]
    df.to_csv(path_or_buf='./mod_step_data.csv', sep=',',index=False)

    for i,df_step in enumerate(list_of_df):

        initial_cond=df_step.iloc[0]
        init_present_pos=initial_cond['Present Position']
        init_time=initial_cond['Time[s]']

        df_step = df_step.iloc[1:].copy()
        df_step['mod_time']=df_step['Time[s]']-init_time
        df_step['mod_present_pos']=df_step['Present Position']-init_present_pos
        df_step['mod_goal_pos']=df_step['Goal Position']-init_present_pos

        plt.plot(df_step['mod_time'],df_step['mod_present_pos'], 'g', linewidth=1.5)
        plt.plot(df_step['mod_time'],df_step['mod_goal_pos'], 'b', linewidth=1.5)
        #plt.plot(df_step['mod_time'],df_step['K*error'], 'y', linewidth=1.5)
        df_step.to_csv(path_or_buf=f'./mod_data_{i}.csv', sep=',',index=False)

        step_amplitude=df_step['mod_goal_pos'].values[0]
        steady_state_position=df_step['mod_present_pos'].values[-1]
        K_open_loop=(step_amplitude-(step_amplitude-steady_state_position))/(Kp*(step_amplitude-steady_state_position))
        K_closed_loop=steady_state_position/step_amplitude
        #K_closed_loop=K_open_loop*Kp/(K_open_loop*Kp+1)
        tau, _ = curve_fit(lambda x, tau: model(x, tau, steady_state_position), df_step['mod_time'].values, df_step['mod_present_pos'].values)
        y=model(df_step['mod_time'], tau[0], steady_state_position)
        plt.plot(df_step['mod_time'],y, 'r', linewidth=1.5)
        plt.show()

        print(i, tau[0], K_closed_loop)
        print(i, tau[0]*(1+Kp*K_open_loop), K_open_loop)



