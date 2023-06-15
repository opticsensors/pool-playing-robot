import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from pool.pool_sim import Params
from pool.random_balls import RandomBalls
from pool.pool_frame import Rectangle
from pool.billiard_env import BilliardEnv
from gym.spaces.utils import flatdim

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

d_centroids={0:(200,103),8:(400,400),9:(500,600),14:(250,103)}
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

states = flatdim(env.observation_space)
actions = env.action_space.n

model = Sequential()  
model.add(Flatten(input_shape=(1,states)))  
model.add(Dense(24, activation='relu', input_shape=states))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy = BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10, 
    target_model_update=0.01)    

agent.compile(Adam(lr=0.001), metrics=['mae'])
agent.fit(env, nb_steps=50000, visualize=False, verbose=1)
results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()