import numpy as np
import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from pool.utils import Params, Rectangle
from pool.billiard_env import BilliardEnv

d_centroids={0:(200,103),8:(400,400),1:(500,500),9:(250,103)}
computation_rectangle = Rectangle((0,0), Params().DISPLAY_SIZE)
computation_rectangle = computation_rectangle.get_rectangle_with_offsets((130, 130, 130, 130))

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
#env=NormalizeReward(env)
#env=NormalizeObservation(env)
#env=RescaleAction(env, 0, 1)

env.reset()
print('obs_sape',[e for e in env.observation_space])
print('state',[e for e in env.state])

model = PPO('MultiInputPolicy', env, verbose=1, 
	    learning_rate=0.00003,
		n_steps=2048, 
		batch_size=64, 
		n_epochs=20
	    )
model = PPO.load('./PP0.zip', env)
print('learning ... ')
model.learn(total_timesteps=5000)
model.save('PP0')

print('testing ...')
episodes = 10
for ep in range(episodes):
	obs, info = env.reset()
	terminate = False
	truncated = False
	print(ep)
	while not terminate and not truncated:
		action, _states = model.predict(obs)
		obs, rewards, terminate, truncate, info = env.step(action)
		env.render()
		print(rewards)

