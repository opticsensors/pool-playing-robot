import random
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
from pool.pool_sim import PhysicsSim
import pygame
from pool.random_balls import RandomBalls
from pool.utils import Params, Rectangle

class BilliardEnv(gym.Env):
  """
  State is composed of:
  s = {turn: 0 or 1, ball_i: [xi, yi], ball_j: [xj, yj], ...}
  """

  def __init__(self, computation_rectangle, d_centroids,cushions,pockets, max_steps=5000, render_mode ='human'):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    self.computation_rectangle = computation_rectangle
    self.min_xy = computation_rectangle.top_left 
    self.max_xy = computation_rectangle.bottom_right

    self.screen = None
    self.clock = None
    self.params = Params()
    self.params.MAX_ENV_STEPS = max_steps
    self.physics_eng = PhysicsSim()
    self.physics_eng.create_balls(d_centroids)
    self.physics_eng.create_cushions(cushions)
    self.physics_eng.create_pockets(pockets)
    self.physics_eng.handle_collisions()

    ## Ball XY positions can be between pool table limits                                   
    dict_spaces={ str(ball_num): spaces.Box(low=np.float32(self.min_xy), 
                                     high=np.float32(self.max_xy), 
                                     shape=(2,), 
                                     dtype=np.float32) for ball_num in d_centroids}
    dict_spaces = {**{'turn': spaces.Discrete(2) }, **dict_spaces}
    self.observation_space = spaces.Dict(dict_spaces)

    #angle between 0 and 3600 degrees
    self.action_space = spaces.Box(low=np.float32(np.array([0])), high=np.float32(np.array([360])), dtype=np.float32)

    self.goals = np.array([hole['pose'] for hole in self.physics_eng.holes])
    self.goalRadius = [hole['radius'] for hole in self.physics_eng.holes]
    self.truncate = False
    self.state={}
    self.render_mode = render_mode
    self.reward = 0

  def reset(self, desired_ball_pose=None, turn=None, seed=None, options=None):
    """
    Function to reset the environment.
    - If param RANDOM_BALL_INIT_POSE is set, the ball appears in a random pose, otherwise it will appear at specific locations
    :return: Initial observation
    """
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    if turn is None:
      self.turn = random.randint(0,1)
    else:
      self.turn = turn

    if desired_ball_pose is None:
      random_balls=RandomBalls(ball_radius=self.params.BALL_RADIUS,
                              computation_rectangle=self.computation_rectangle)
      init_ball_pose = random_balls.generate_random_positions_given_balls([0,8,1,9]) # TODO add more balls
    else:
      init_ball_pose = desired_ball_pose.copy()

    # we wont check this cnsition for now because we suppose the random generated data is within pool frame limits
    #for ball_pose in init_ball_pose.values():
    #  ball_pose=np.array(ball_pose)
    #  if (ball_pose < self.min_xy).all() or (ball_pose > self.max_xy).all():
    #    print('truncated')
    #    self.truncate = True

    self.physics_eng.reset(init_ball_pose)
    self.steps = 0
    self.reward = 0
    #self.states is created when we call self._get_obs() 
    observation = self._get_obs()
    info = self._get_info()
    return observation, info

  def _get_obs(self):
    """
    This function returns the state after reading the simulator parameters.
    :return: state: 
    """
    balls_pose={}
    for ball_num, ball_shape in self.physics_eng.balls.items():
      ball_position = np.array(ball_shape.body.position)
      balls_pose[str(ball_num)] = ball_position

    self.state=(balls_pose).copy()
    return {**{'turn': self.turn }, **self.state}
  
  def _get_info(self):
    return {"turn": self.turn,
            "termination_positions": self.state,
            "total_collisions": self.physics_eng.total_collisions,
            "first_ball_collision": self.physics_eng.first_ball_collision,
            }

  def reward_function(self, info):
    """
    This function calculates the reward
    :return:
    """
    done = False
    #check if any balls have been potted
    for ball_num, ball_position in self.state.items():
      distances = np.linalg.norm(ball_position - self.goals, axis=1)
      if (distances <= self.goalRadius).any():
        done = True
        if ball_num in ['1','2','3','4','5','6','7'] and self.turn==0: #solid
          self.reward += 1
          info['potted_ball']='correct_ball'
        elif ball_num in ['9','10','11','12','13','14','15'] and self.turn==1: #strip
          self.reward += 1
          info['potted_ball']='correct_ball'
        else:
          info['potted_ball']='wrong_ball'

    if all(abs(ball_shape.body.velocity) <= self.params.BALL_TERMINAL_VELOCITY 
           for ball_shape in self.physics_eng.balls.values()):
      done = True

    return self.reward, done, info

  def step(self, action):
    """
    Performs an environment step.
    :return: state, reward, final, info
    """
    self.steps += 1
    self.physics_eng.move_cue_ball(action)
    done = False
    while not done:
      ## Simulate timestep
      self.physics_eng.step()
      ## Get state
      observation = self._get_obs()
      info = {}
      # Get reward
      reward, done, info = self.reward_function(info)
      #self.render() # TODO remove

    if self.steps >= self.params.MAX_ENV_STEPS:  ## Check if max number of steps has been exceeded
      done = True

    info = {**info, **self._get_info()}
    info['action'] = action
    if 'potted_ball' not in info:
      info['potted_ball'] = 'none'

    return observation, reward, done, False, info

  def render(self):
    """
    Rendering function
    :param mode: if human, renders on screen. If rgb_array, renders as numpy array
    :return: screen if mode=human, array if mode=rgb_array
    """
    # If no screen available create screen

    if self.screen is None and self.render_mode == "human":
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]))
        pygame.display.set_caption('Billiard')

    if self.clock is None and self.render_mode == "human":
        self.clock = pygame.time.Clock()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
          self.close()

    canvas = pygame.Surface((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]))
    canvas.fill((75, 75, 75))

    # First we draw the cushions
    for c in self.physics_eng.cushions:
      pygame.draw.polygon(
          canvas,
          (0, 100, 0),
          c,
          )
    # Now we draw the agent
    pygame.draw.circle(
        canvas,
        (255, 255, 255),
        (self.state['0']),
        self.params.BALL_RADIUS,
    )

    # draw 8ball
    pygame.draw.circle(
        canvas,
        (25, 25, 25),
        (self.state['8']),
        self.params.BALL_RADIUS,
    )

    for goal, radius in zip(self.goals, self.goalRadius):
      pygame.draw.circle(
        canvas,
        (0, 0, 0),
        (goal),
        radius,
    )

    # Finally, add other balls
    for ball_num in self.state:
      if ball_num in ['1','2','3','4','5','6','7']:
        pygame.draw.circle(
          canvas,
          (100, 149, 237),
          (self.state[ball_num]),
          self.params.BALL_RADIUS,
        )
      elif ball_num in ['9','10','11','12','13','14','15']:
        pygame.draw.circle(
          canvas,
          (255, 129, 0),
          (self.state[ball_num]),
          self.params.BALL_RADIUS,
        )

    if self.render_mode == "human":
        # The following line copies our drawings from `canvas` to the visible window
        self.screen.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.params.TARGET_FPS)
    else:  # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
  def close(self):
    if self.screen is not None:
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
  params=Params()
  d_centroids={0: (414, 248), 8: (543, 265), 1: (305, 328), 9: (480, 346)}
  computation_rectangle = Rectangle((0,0), Params().DISPLAY_SIZE)
  computation_rectangle = computation_rectangle.get_rectangle_with_offsets((127, 127, 127, 127))
  env=BilliardEnv(computation_rectangle,d_centroids, params.CUSHIONS, params.POCKETS)
  
  #print(env.action_space.sample())
  #print(env.observation_space.sample())
  
  observation, info = env.reset(d_centroids,0)
  import time
  for i in range(10):
    #action = env.action_space.sample()
    action=149
    observation, reward, terminated, truncated, info = env.step(action)
    print(info['potted_ball'])
    #env.render()
    time.sleep(1)
    if terminated or truncated:
        print('reseting', terminated, truncated)
        observation, info = env.reset(d_centroids,0)
  env.close()