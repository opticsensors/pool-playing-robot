from pool.pool_env import PoolEnv
from pool.utils import Params

params=Params()
d_centroids={0: (3138, 2279), 8: (332, 739), 2: (2375, 815), 3: (3340, 1821), 4: (2880, 601), 1: (3587, 1719), 6: (3212, 653), 7: (2820, 2358), 14: (2386, 1562), 13: (2495, 2303), 9: (1284, 1282), 10: (1997, 1464), 11: (934, 925), 15: (286, 
1598)}
env=PoolEnv(params.COMPUTATIONAL_RECTANGLE,d_centroids, params.CUSHIONS, params.POCKETS, render_mode = 'human')
turn = 0

observation, info = env.reset(d_centroids, turn)
import time
for i in range(10):
    #action = env.action_space.sample()
    action=170.4
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(1)
    if terminated or truncated:
        observation, info = env.reset(d_centroids, turn)
env.close()