import time
import pandas as pd
from pool.utils import Params, Rectangle
from pool.billiard_env_v4 import BilliardEnv
from pool.random_balls import RandomBalls

params=Params()
computation_rectangle = Rectangle((0,0), Params().DISPLAY_SIZE)
computation_rectangle = computation_rectangle.get_rectangle_with_offsets((127, 127, 127, 127))

l_configs=[]
random_balls=RandomBalls(ball_radius=params.BALL_RADIUS, computation_rectangle=computation_rectangle)
for _ in range(5):
    l_configs.append(random_balls.generate_random_positions_given_balls([0,8,1,9]))

env=BilliardEnv(computation_rectangle, l_configs[0], params.CUSHIONS, params.POCKETS)
observation, info = env.reset(l_configs[0])

incr=0.25
dict_to_save = {}
list_of_dict = []
for i,config in enumerate(l_configs):
    action = 0
    while action<360:
        observation, reward, terminated, truncated, info = env.step(action)
        print(i,action)
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
            observation, info = env.reset(config)

        action += incr

    df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
    df.to_csv(path_or_buf=f'./pool_sim_data.csv', sep=',',index=False, mode='a', header=False)
    dict_to_save = {}
    list_of_dict = []

env.close()

#df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
#df.to_csv(path_or_buf=f'./pool_sim_data.csv', sep=',',index=False)