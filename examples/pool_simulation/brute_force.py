import time
import numpy as np
import pandas as pd
from pool.utils import Params, Rectangle
from pool.billiard_env_single_shot import BilliardEnv
from pool.random_balls import RandomBalls

params=Params()
computation_rectangle = Rectangle((0,0), Params().DISPLAY_SIZE)
computation_rectangle = computation_rectangle.get_rectangle_with_offsets((127, 127, 127, 127))

random_balls=RandomBalls(ball_radius=params.BALL_RADIUS, computation_rectangle=computation_rectangle)
config=random_balls.generate_random_positions_given_balls([0,8,1,2,3,4,5,6,11,12,13,14])
#config={0: (832, 420), 8: (617, 502), 1: (247, 308), 9: (472, 501)}
env=BilliardEnv(computation_rectangle, config, params.CUSHIONS, params.POCKETS)
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
            dict_to_save['potted_ball']=info['potted_ball']
            dict_to_save['action']=info['action']

            # add initial positions
            for ball_num,ball in config.items():
                dict_to_save[f'start_ball{ball_num}_x']=ball[0]
                dict_to_save[f'start_ball{ball_num}_y']=ball[1]

            # add final positions
            for ball_num,ball in info['termination_positions'].items():
                dict_to_save[f'end_ball{ball_num}_x']=ball[0]
                dict_to_save[f'end_ball{ball_num}_y']=ball[1]

            list_of_dict.append(dict_to_save.copy())
            env.close()
            env=BilliardEnv(computation_rectangle, config, params.CUSHIONS, params.POCKETS)
            observation, info = env.reset(config, turn)

    df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
    df.to_csv(path_or_buf=f'./brute_force_data.csv', sep=',',index=False)
    env.close()
    return df

first_sweep = list(np.arange(0, 360, 2))
df1 = angle_sweep(env, first_sweep)
valid_actions = df1['action'][df1['potted_ball']=='correct_ball']
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
valid_actions = df2['action'][df2['potted_ball']=='correct_ball']

print(config)
print(valid_actions)