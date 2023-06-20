import numpy as np
import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from pool.utils import Params, Rectangle
from pool.billiard_env_v3 import BilliardEnv
#from stable_baselines3.common.noise import NormalActionNoise

params = Params()
d_centroids={0:(200,103),8:(400,400),1:(500,500),9:(250,103)}
computation_rectangle = params.COMPUTATIONAL_RECTANGLE
pockets = params.POCKETS
cushions = params.CUSHIONS
env=BilliardEnv(computation_rectangle,d_centroids, cushions, pockets)
#env=NormalizeReward(env)
#env=NormalizeObservation(env)
#env=RescaleAction(env, 0, 1)
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

env.reset()

model = PPO('MultiInputPolicy', env, verbose=1, 
	    learning_rate=0.0003,
		n_steps=2048, 
		batch_size=64, 
		n_epochs=10
	    )
model = PPO.load('./PP0.zip', env)
model.set_env(env)
#print('learning ... ')
#model.learn(total_timesteps=10)
#model.save('PP0')

print('testing ...')
episodes = 5
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

