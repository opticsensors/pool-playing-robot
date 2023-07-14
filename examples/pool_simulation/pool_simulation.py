from pool.pool_env import PoolEnv
from pool.utils import Params

params=Params()
d_centroids={0: (414, 248), 8: (543, 265), 1: (305, 328), 9: (480, 346)}
env=PoolEnv(params.COMPUTATIONAL_RECTANGLE,d_centroids, params.CUSHIONS, params.POCKETS, render_mode = 'human')

observation, info = env.reset()
import time
for i in range(10):
    action = env.action_space.sample()
    #action=149
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(1)
    if terminated or truncated:
        observation, info = env.reset()
env.close()