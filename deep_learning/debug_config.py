import time
import pandas as pd
from pool.utils import Params, Rectangle
from pool.billiard_env import BilliardEnv

params=Params()

#manual setting
d_centroids={0:(192,399),8:(492,261),9:(983,345),1:(905,285)}
action=319.3520000000031
turn=1

#df setting
df = pd.read_csv('./pool_sim_data.csv', sep=',',decimal='.')
df_correct=df[df['potted_ball']=='correct_ball']
col_names=df_correct.columns.tolist()
starting_positions = [col_name for col_name in col_names if col_name.startswith('start')]
#knowing that pairs of columns go together:
for starting_position_x, starting_position_y in zip(starting_positions[0::2], starting_positions[1::2]):
    ball_num=starting_position_x.split('_')[-2]
    coord_x = df_correct[starting_position_x]
    coord_y = df_correct[starting_position_y]
    tuple_of_coord = list(zip(coord_x, coord_y))
    df_correct[int(ball_num)] = tuple_of_coord

actions=df_correct['action'].tolist()
turns=df_correct['turn'].tolist()
d_configs=df_correct.drop(columns=col_names).to_dict(orient='records')

i=108
d_centroids=d_configs[i]
action=actions[i]
turn=turns[i]

computation_rectangle = Rectangle((0,0), Params().DISPLAY_SIZE)
computation_rectangle = computation_rectangle.get_rectangle_with_offsets((127, 127, 127, 127))
env=BilliardEnv(computation_rectangle,d_centroids, params.CUSHIONS, params.POCKETS)
observation, info = env.reset(d_centroids, turn)

for i in range(1000):
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.02)
    if terminated or truncated:
        observation, info = env.reset(d_centroids, turn)
env.close()