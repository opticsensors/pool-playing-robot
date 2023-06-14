import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from pool.pool_sim import PhysicsSim, Params
import pygame
from pool.random_balls import RandomBalls

class BilliardEnv(gym.Env):
  """
  State is composed of:
  s = ([ball_x, ball_y])

  The values that these components can take are:
  ball_x, ball_y -> [-1.5, 1.5]
  """
  metadata = {'render.modes': ['human'],
              'video.frames_per_second': 15
              }

  def __init__(self, computation_rectangle, d_centroids, seed=None, max_steps=500):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    self.d_centroids = d_centroids
    self.computation_rectangle = computation_rectangle
    self.min_xy = computation_rectangle.top_left
    self.max_xy = computation_rectangle.top_left

    self.screen = None
    self.params = Params()
    self.params.MAX_ENV_STEPS = max_steps
    self.physics_eng = PhysicsSim()

    ## Ball XY positions can be between -1.5 and 1.5
    #min_ball_coord = np.repeat( self.min_xy, len(self.d_centroids), axis=0)
    #max_ball_coord = np.repeat( self.max_xy, len(self.d_centroids), axis=0)
    #self.observation_space = spaces.Box(low=min_ball_coord,high=max_ball_coord, dtype=np.float32)                                       
    
    self.observation_space = spaces.Dict(
            {
                ball_num: spaces.Box(low=self.min_xy, high=self.max_xy, shape=(2,), dtype=np.float32)
                for ball_num in self.d_centroids.keys()
            }
    )

    ## Joint commands can be between [-1, 1]
    self.action_space = spaces.Box(low=np.array([0]), high=np.array([360]), dtype=np.float32)

    self.goals = np.array([hole['pose'] for hole in self.physics_eng.holes])
    self.goalRadius = [hole['radius'] for hole in self.physics_eng.holes]
    self.rew_area = None

    self.seed(seed)

  def seed(self, seed=None):
    """
    Function to seed the environment
    :param seed: The random seed
    :return: [seed]
    """
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self, desired_ball_pose = None):
    """
    Function to reset the environment.
    - If param RANDOM_BALL_INIT_POSE is set, the ball appears in a random pose, otherwise it will appear at [-0.5, 0.2]
    :return: Initial observation
    """
    if desired_ball_pose is None:
      random_balls=RandomBalls(ball_radius=self.params.BALL_RADIUS,
                              computation_rectangle=self.computation_rectangle)
      init_ball_pose = random_balls.generate_random_balls()
      #init_ball_pose = {k: np.array([v]) for k, v in desired_ball_pose.items()}
    else:
      init_ball_pose = desired_ball_pose.copy()

    self.physics_eng.reset(init_ball_pose)
    self.steps = 0
    self.rew_area = None
    return self._get_obs()

  def _get_obs(self):
    """
    This function returns the state after reading the simulator parameters.
    :return: state: composed of ([ball_pose_x, ball_pose_y])
    """
    ball_pose = self.physics_eng.balls[0].position + self.physics_eng.wt_transform

    if np.abs(ball_pose[0]) > 1.5 or np.abs(ball_pose[1]) > 1.5:
      raise ValueError('Ball out of map in position: {}'.format(ball_pose))

    self.state = np.array([ball_pose[0], ball_pose[1]])
    return self.state

  def reward_function(self, info):
    """
    This function calculates the reward
    :return:
    """
    ball_pose = self.state[0:2]
    for hole_idx, hole in enumerate(self.physics_eng.holes):
      dist = np.linalg.norm(ball_pose - hole['pose'])
      if dist <= hole['radius']:
        done = True
        reward = 100
        info['reason'] = 'Ball in hole'
        info['rew_area'] = hole_idx
        return reward, done, info
    return 0, False, None

  def step(self, action):
    """
    Performs an environment step.
    :param action: Arm Motor commands. Can be either torques or velocity, according to TORQUE_CONTROL parameter
    :return: state, reward, final, info
    """
    # action = np.clip(action, -1, 1)

    self.steps += 1
    ## Pass motor command
    self.physics_eng.move_joint('jointW0', action[0])
    self.physics_eng.move_joint('joint01', action[1])
    ## Simulate timestep
    self.physics_eng.step()
    ## Get state
    self._get_obs()
    info = {}

    # Get reward
    reward, done, info = self.reward_function(info)

    if self.steps >= self.params.MAX_ENV_STEPS:  ## Check if max number of steps has been exceeded
      done = True
      info['reason'] = 'Max Steps reached: {}'.format(self.steps)

    return self.state, reward, done, info

  def render(self, mode='rgb_array', **kwargs):
    """
    Rendering function
    :param mode: if human, renders on screen. If rgb_array, renders as numpy array
    :return: screen if mode=human, array if mode=rgb_array
    """
    # If no screen available create screen
    if self.screen is None and mode == 'human':
      self.screen = pygame.display.set_mode((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]), 0, 32)
      pygame.display.set_caption('Billiard')
      self.clock = pygame.time.Clock()

    if self.state is None: return None ## If there is no state, exit

    if mode == 'human':
      self.screen.fill(pygame.color.THECOLORS["white"]) ## Draw white background
    elif mode == 'rgb_array':
      capture = pygame.Surface((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]))
      capture.set_alpha(None)

    ## Draw holes. This are just drawn, but are not simulated.
    for goal, radius in zip(self.goals, self.goalRadius):
      ## To world transform (The - is to take into account pygame coordinate system)
      pose = -goal + self.physics_eng.tw_transform

      if mode == 'human':
        ## Draw the holes on the screen
        pygame.draw.circle(self.screen,
                           (255, 0, 0),
                           [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                           int(radius * self.params.PPM))
      elif mode == 'rgb_array':
        ## Draw the holes on the capture
        pygame.draw.circle(capture,
                           (255, 0, 0),
                           [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                           int(radius * self.params.PPM))

    ## Draw bodies
    for body in self.physics_eng.world.bodies:
      color = [0, 0, 0]
      obj_name = body.userData['name']
      if obj_name == 'ball0':
        color = [0, 0, 255]
      elif obj_name in ['link0', 'link1']:
        color = [100, 100, 100]
      elif 'wall' in obj_name:
        color = [150, 150, 150]

      for fixture in body.fixtures:
        if mode == 'human':
          fixture.shape.draw(body, self.screen, self.params, color)
        elif mode == 'rgb_array':
          obj_name = body.userData['name']
          if self.params.SHOW_ARM_IN_ARRAY: ## If param is set, in the rgb_array the arm will be visible
            fixture.shape.draw(body, capture, self.params, color)
          else:
            if not obj_name in ['link0', 'link1']:
              fixture.shape.draw(body, capture, self.params, color)

    if mode == 'human':
      pygame.display.flip() ## Need to flip cause of drawing reasons
      self.clock.tick(self.params.TARGET_FPS)
      return self.screen
    elif mode == 'rgb_array':
      imgdata = pygame.surfarray.array3d(capture)
      return imgdata.swapaxes(0, 1)