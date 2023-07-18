from pool.pool_env import PoolEnv
from pool.utils import Params

params=Params()
env=PoolEnv(render_mode = 'human')
#d_centroids={0: (3274, 849), 8: (1690, 244), 1: (2412, 1670), 12: (674, 500)}
#turn = 0

observation, info = env.reset()
import time
for i in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(1)
    if terminated or truncated:
        observation, info = env.reset()
env.close()