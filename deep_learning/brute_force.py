from pool.pool_frame import Rectangle
from pool.pool_sim import Params
from pool.billiard_env import BilliardEnv
import time

d_centroids={0:(200,103),8:(400,400),9:(500,500),14:(250,103)}
computation_rectangle = Rectangle((0,0), Params().DISPLAY_SIZE)
computation_rectangle = computation_rectangle.get_rectangle_with_offsets((127, 127, 127, 127))


#create six pockets on table
pockets = [
(55, 63),
(592, 48),
(1134, 64),
(55, 616),
(592, 629),
(1134, 616)
]

#create pool table cushions
cushions = [
[(88, 56), (109, 77), (555, 77), (564, 56)],
[(621, 56), (630, 77), (1081, 77), (1102, 56)],
[(89, 621), (110, 600),(556, 600), (564, 621)],
[(622, 621), (630, 600), (1081, 600), (1102, 621)],
[(56, 96), (77, 117), (77, 560), (56, 581)],
[(1143, 96), (1122, 117), (1122, 560), (1143, 581)]]

env=BilliardEnv(computation_rectangle,d_centroids, cushions, pockets)

#print(env.action_space.sample())
#print(env.observation_space.sample())
gen=0
incr=10
action=0
terminated=False
truncated=False
observation, info = env.reset(d_centroids, 1)
while action<360:
    gen+=1
    while not terminated or truncated:
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)
    print(gen,action, reward, info)
    action += incr
    terminated=False
    truncated=False
    observation, info = env.reset(d_centroids, 1)

env.close()
