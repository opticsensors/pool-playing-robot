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
      init_ball_pose = random_balls.generate_random_positions_given_balls([0,8,1,9])
    else:
      init_ball_pose = desired_ball_pose.copy()

    for ball_pose in init_ball_pose.values():
      ball_pose=np.array(ball_pose)
      if (ball_pose < self.min_xy).all() or (ball_pose > self.max_xy).all():
        print('truncated')
        self.truncate = True

    self.physics_eng.reset(init_ball_pose)
    self.steps = 0
    self.reward = 0
    #self.states is created when we call self._get_obs() 
    observation = self._get_obs()
    return observation, self._get_info()

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
            "total_collisions": self.physics_eng.total_collisions,
            "ball_collision_happened": self.physics_eng.ball_collision_happened,
            "first_ball_collision": self.physics_eng.first_ball_collision
            }

  def reward_function(self, info):
    """
    This function calculates the reward
    :return:
    """
    done = False
    potted_balls = []
    info['turn']=self.turn
    #check if any balls have been potted
    for ball_num, ball_position in self.state.items():
      distances = np.linalg.norm(ball_position - self.goals, axis=1)
      if (distances <= self.goalRadius).any():
        potted_balls.append(ball_num)
        done = True
        info['reason'] = 'Terminated: ball in hole'
        if ball_num in ['1','2','3','4','5','6','7'] and self.turn==0: #solid
          self.reward += 50
          info['pot solid']=ball_num
        elif ball_num in ['9','10','11','12','13','14','15'] and self.turn==1: #strip
          self.reward += 50
          info['pot strip']=ball_num
        else:
          self.reward += -10
          info['pot wrong ball']=ball_num
    #info['potted_balls'] = potted_balls

    if all(abs(ball_shape.body.velocity) <= self.params.BALL_TERMINAL_VELOCITY 
           for ball_shape in self.physics_eng.balls.values()):
      done = True
      info['reason'] = 'Terminated: balls velocity 0'
    
    if self.physics_eng.total_collisions > 6:
      self.reward += -1
      info['invalid num coll']=self.physics_eng.total_collisions

    if self.physics_eng.ball_collision_happened:
      if self.turn == self.physics_eng.first_ball_collision:
        self.reward += 4
      else:
        self.reward += -1
      self.physics_eng.ball_collision_happened = False # so we dont add this multiple times

    return self.reward, done, info

  def step(self, action):
    """
    Performs an environment step.
    :return: state, reward, final, info
    """
    # apply action only in the begining 
    if self.steps == 0:
      self.physics_eng.move_cue_ball(action)

    self.steps += 1
    ## Simulate timestep
    self.physics_eng.step()
    ## Get state
    observation = self._get_obs()
    info = {}

    # Get reward
    reward, done, info = self.reward_function(info)

    if self.steps >= self.params.MAX_ENV_STEPS:  ## Check if max number of steps has been exceeded
      print('terminated in max steps')
      done = True
      info['reason'] = 'Max Steps reached: {}'.format(self.steps)

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
  
  d_centroids={0:(200,103),8:(400,400),9:(500,600),1:(250,103)}
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
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print('step',i,'turn',env.turn,
          'total_coll',env.physics_eng.total_collisions,
          'is_ball_coll',env.physics_eng.ball_collision_happened, 
          'first_coll',env.physics_eng.first_ball_collision,
          'rew', reward)
    env.render()
    time.sleep(0.05)
    if terminated or truncated:
        print('reseting', terminated, truncated)
        observation, info = env.reset()
  env.close()