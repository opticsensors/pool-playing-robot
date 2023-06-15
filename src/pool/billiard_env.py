import random
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
from pool.pool_sim import PhysicsSim, Params
import pygame
from pool.random_balls import RandomBalls
from pool.pool_frame import Rectangle

class BilliardEnv(gym.Env):
  """
  State is composed of:
  s = ([ball_x, ball_y])

  The values that these components can take are:
  ball_x, ball_y -> [-1.5, 1.5]
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

    ## Ball XY positions can be between -1.5 and 1.5
    #min_ball_coord = np.repeat( self.min_xy, len(d_centroids), axis=0)
    #max_ball_coord = np.repeat( self.max_xy, len(d_centroids), axis=0)
    #self.observation_space = spaces.Box(low=min_ball_coord,high=max_ball_coord, dtype=np.float32)                                       
    
    dict_spaces={ str(ball_num): spaces.Box(low=np.float32(self.min_xy), 
                                     high=np.float32(self.max_xy), 
                                     shape=(2,), 
                                     dtype=np.float32) for ball_num in d_centroids}
    dict_spaces = {**{'turn': spaces.Discrete(2) }, **dict_spaces}

    self.observation_space = spaces.Dict(dict_spaces)

    ## Joint commands can be between [-1, 1]
    self.action_space = spaces.Box(low=np.float32(np.array([0])), high=np.float32(np.array([360])), dtype=np.float32)

    self.goals = np.array([hole['pose'] for hole in self.physics_eng.holes])
    self.goalRadius = [hole['radius'] for hole in self.physics_eng.holes]
    self.truncate = False

    self.render_mode = render_mode

  def reset(self, desired_ball_pose = None, seed=None, options=None):
    """
    Function to reset the environment.
    - If param RANDOM_BALL_INIT_POSE is set, the ball appears in a random pose, otherwise it will appear at [-0.5, 0.2]
    :return: Initial observation
    """
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    if desired_ball_pose is None:
      random_balls=RandomBalls(ball_radius=self.params.BALL_RADIUS,
                              computation_rectangle=self.computation_rectangle)
      init_ball_pose = random_balls.generate_random_balls()
    else:
      init_ball_pose = desired_ball_pose.copy()

    self.physics_eng.reset(init_ball_pose)
    self.steps = 0
    self.turn = random.randint(0,1)

    dict_spaces={ str(ball_num): spaces.Box(low=np.float32(self.min_xy), 
                                     high=np.float32(self.max_xy), 
                                     shape=(2,), 
                                     dtype=np.float32) for ball_num in init_ball_pose}
    dict_spaces = {**{'turn': spaces.Discrete(2) }, **dict_spaces}
    self.observation_space = spaces.Dict(dict_spaces)

    return self._get_obs(), self._get_info()

  def _get_obs(self):
    """
    This function returns the state after reading the simulator parameters.
    :return: state: composed of ([ball_pose_x, ball_pose_y])
    """
    balls_pose={}
    for ball_num, ball_shape in self.physics_eng.balls.items():
      ball_position = np.array(ball_shape.body.position)
      balls_pose[str(ball_num)] = ball_position

      if (ball_position < self.min_xy).all() or (ball_position > self.max_xy).all():
        print('truncated')
        self.truncate = True
    self.state = balls_pose
    return {**{'turn': self.turn }, **self.state}
  
  def _get_info(self):
    return {"turn": self.turn,
            "total_collisions": self.physics_eng.total_collisions,
            "accumulative_angles": self.physics_eng.accumulative_angles,
            "collision_occurs": self.physics_eng.collision_occurs,
            }

  def reward_function(self, info):
    """
    This function calculates the reward
    :return:
    """
    reward = 0
    done = False
    potted_balls = []
    #check if any balls have been potted
    for ball_num, ball_position in self.state.items():
      distances = np.linalg.norm(ball_position - self.goals, axis=1)
      if (distances <= self.goalRadius).any():
        potted_balls.append(ball_num)
        print('terminated in dist to pockets')
        done = True
        info['reason'] = 'Ball in hole'
        if ball_num in [1,2,3,4,5,6,7] and self.turn==0: #solid
          reward = 100
        elif ball_num in [9,10,11,12,13,14,15] and self.turn==1: #strip
          reward = 100
        else:
          reward = -100
    info['potted_balls'] = potted_balls

    if all(abs(ball_shape.body.velocity) <= self.params.BALL_TERMINAL_VELOCITY 
           for ball_shape in self.physics_eng.balls.values()):
      print('terminated in velocity')
      done = True
      info['reason'] = 'Balls stopped moving'
    
    if self.physics_eng.total_collisions > 5 or self.physics_eng.total_collisions == 0:
      reward = -50
    if self.physics_eng.collision_occurs:
      if abs(self.physics_eng.angle) < 60:
        reward = 25

    return reward, done, info

  def step(self, action):
    """
    Performs an environment step.
    :param action: Arm Motor commands. Can be either torques or velocity, according to TORQUE_CONTROL parameter
    :return: state, reward, final, info
    """
    # apply action only in the begining 
    if self.steps == 0:
      self.physics_eng.move_cue_ball(action)

    self.steps += 1
    ## Simulate timestep
    self.physics_eng.step()
    ## Get state
    self._get_obs()
    info = {}

    # Get reward
    reward, done, info = self.reward_function(info)

    if self.steps >= self.params.MAX_ENV_STEPS:  ## Check if max number of steps has been exceeded
      print('terminated in max steps')
      done = True
      info['reason'] = 'Max Steps reached: {}'.format(self.steps)

    return {**{'turn': self.turn }, **self.state}, reward, done, False, info

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
  
  #print(env.action_space.sample())
  #print(env.observation_space.sample())
  
  observation, info = env.reset()
  import time
  for i in range(1000):
    print(i)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.01)
    if terminated or truncated:
        print('reseting', terminated, truncated)
        observation, info = env.reset()
  env.close()