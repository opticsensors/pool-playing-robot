import numpy as np
import pandas as pd
from pool.utils import Params
from pool.pool_env import PoolEnv
from pool.random_balls import RandomBalls

params=Params()

random_balls=RandomBalls(ball_radius=params.BALL_RADIUS, computation_rectangle=params.COMPUTATIONAL_RECTANGLE)
config=random_balls.generate_random_balls()
#config={0: (832, 420), 8: (617, 502), 1: (247, 308), 9: (472, 501)}
env=PoolEnv(params.COMPUTATIONAL_RECTANGLE, config, params.CUSHIONS, params.POCKETS, render_mode = None)
turn = 0

observation, info = env.reset(config, turn)

def angle_sweep(env, angles_to_study):
    dict_to_save = {}
    list_of_dict = []
    for action in angles_to_study:
        observation, reward, terminated, truncated, info = env.step(action)
        print(action)
        if terminated or truncated:
            dict_to_save['turn']=info['turn']
            dict_to_save['total_collisions']=info['total_collisions']
            dict_to_save['first_ball_collision']=info['first_ball_collision']
            if info['potted_ball'] == 'correct_ball':
                dict_to_save['potted_ball'] = 1
            else:
                dict_to_save['potted_ball'] = 0
            dict_to_save['action']=info['action']

            list_of_dict.append(dict_to_save.copy())
            env.close()
            env=PoolEnv(params.COMPUTATIONAL_RECTANGLE, config, params.CUSHIONS, params.POCKETS, render_mode = None)
            observation, info = env.reset(config, turn)

    df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
    env.close()
    return df

first_sweep = list(np.arange(0, 360, 2))
df1 = angle_sweep(env, first_sweep)
#valid_actions = df1['action'][df1['potted_ball']=='correct_ball']
possible_valid_actions = df1['action'][df1['first_ball_collision']==turn]
range_width=0.4
possible_valid_ranges = []
for action in possible_valid_actions:
    action_range = (action-range_width, action+range_width)
    possible_valid_ranges.append(action_range)

result = []
for item in sorted(possible_valid_ranges):
    result = result or [item]
    if item[0] > result[-1][1]:
        result.append(item)
    else:
        old = result[-1]
        result[-1] = (old[0], max(old[1], item[1]))

second_sweep = []
for angle_range in result:
    second_sweep.extend(list(np.arange(angle_range[0], angle_range[1],0.1)))

df2 = angle_sweep(env, second_sweep)
#valid_actions = df2['action'][df2['potted_ball']=='correct_ball']

def centroid(data):
    centroids=[]
    weights=[]
    prev_val=0
    for i,value in enumerate(data):
        #from 0 to 1
        if value>prev_val:
            prev_val=value
            i_ini=i
        
        #from 1 to 0
        elif value < prev_val:
            l=i-i_ini
            centroid=(i+i_ini)//2
            weights.append(l)
            centroids.append(centroid)
            prev_val=value
            
    arr_cetroids=np.array(centroids)
    arr_weights=np.array(weights)
    
    #plt.plot(arr_cetroids,arr_weights)
    maximum=arr_cetroids[np.unravel_index(np.nanargmax(arr_weights, axis=None), arr_weights.shape)]
    
    return maximum

max_id=centroid(df2['potted_ball'].to_list())
angle=df2['action'].iloc[max_id]
print('angle', angle)
print('config', config)

