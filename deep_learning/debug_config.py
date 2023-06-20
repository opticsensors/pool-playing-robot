import time
import pandas as pd
from pool.utils import Params, Rectangle
from pool.billiard_env import BilliardEnv

params=Params()
d_centroids={0:(988,491),8:(211,468),9:(1032,541),1:(696,484)}
action=75
turn=1
computation_rectangle = Rectangle((0,0), Params().DISPLAY_SIZE)
computation_rectangle = computation_rectangle.get_rectangle_with_offsets((127, 127, 127, 127))
env=BilliardEnv(computation_rectangle,d_centroids, params.CUSHIONS, params.POCKETS)
observation, info = env.reset(d_centroids, turn)

for i in range(1000):
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.01)
    if terminated or truncated:
        observation, info = env.reset(d_centroids, turn)
env.close()