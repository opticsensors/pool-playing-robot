import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np

class Params(object):
  """
  Define simulation parameters.
  The world is centered at the lower left corner of the table.
  """
  def __init__(self):
    """
    Constructor
    """
    self.TABLE_SIZE = np.array([1200, 678])
    self.TABLE_CENTER = np.array(self.TABLE_SIZE / 2)
    self.DISPLAY_SIZE = (1200, 678)
    self.TO_PIXEL = np.array(self.DISPLAY_SIZE) / self.TABLE_SIZE

    self.POCKET_CORNER_RADIUS = 25
    self.POCKET_MIDDLE_RADIUS = 25

    self.BALL_RADIUS = 25
    self.BALL_ELASTICITY = 0.8
    self.BALL_FRICTION = 1000
    self.BALL_MASS = 5

    self.CUSHION_ELASTICITY = 0.8

    self.CUE_FORCE = 2000

  # Graphic params
    self.PPM = int(min(self.DISPLAY_SIZE)/max(self.TABLE_SIZE))
    self.TARGET_FPS = 120
    self.TIME_STEP = 1.0 / self.TARGET_FPS

    self.MAX_ENV_STEPS = 300

    self.TORQUE_CONTROL = False
    self.TEST = True

    self.RANDOM_ARM_INIT_POSE = False
    self.RANDOM_BALL_INIT_POSE = False

    self.SHOW_ARM_IN_ARRAY = True


class PhysicsSim(object):
  """
  Physics simulator
  """
  def __init__(self, params=None):
    """
    Constructor
    :param balls_pose: Initial ball poses. Is a list of the ball poses [ball0, ball1, ...]
    :param params: Parameters
    """
    if params is None:
      self.params = Params()
    else:
      self.params = params

    ## Physic simulator
    self.space = pymunk.Space()
    self.static_body = self.space.static_body
    self.dt = self.params.TIME_STEP

  def create_cushions(self,cushions):
    """
    Creates the walls of the table
    :return:
    """
    self.cushions = []
    for c in cushions:
        body = pymunk.Body(body_type = pymunk.Body.STATIC)
        body.position = ((0, 0))
        shape = pymunk.Poly(body, c)
        shape.elasticity = self.params.CUSHION_ELASTICITY
        self.space.add(body, shape)
        self.cushions.append(c)

  def create_balls(self, balls_pose):
    """
    Creates the balls in the simulation at the given positions
    :param balls_pose: Initial pose of the ball in table RF
    :return:
    """
    ## List of balls in simulation
    self.balls = []
    self.balls_numbers = []

    for idx, pose in enumerate(balls_pose):
        mass = self.params.BALL_MASS
        radius = self.params.BALL_RADIUS
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        body = pymunk.Body(mass, inertia)
    
        #body = pymunk.Body()
        body.position = pose
        shape = pymunk.Circle(body, self.params.BALL_RADIUS)
        #shape.mass = self.params.BALL_MASS
        shape.elasticity = self.params.BALL_ELASTICITY
        #use pivot joint to add friction
        pivot = pymunk.PivotJoint(self.static_body, body, (0, 0), (0, 0))
        pivot.max_bias = 0 # disable joint correction
        pivot.max_force = self.params.BALL_FRICTION # emulate linear friction
        self.space.add(body, shape, pivot)
        self.balls.append(shape)
        self.balls_numbers.append(idx)

  def create_pockets(self, pockets):
    """
    Defines the holes in table RF. This ones are not simulated, but just defined as a list of dicts.
    :return:
    """
    # Holes in simulation. Represented as list of dicts.
    self.holes = [{'pose': np.array(pockets[0]), 'radius': self.params.POCKET_CORNER_RADIUS},
                  {'pose': np.array(pockets[1]), 'radius': self.params.POCKET_MIDDLE_RADIUS},
                  {'pose': np.array(pockets[2]), 'radius': self.params.POCKET_CORNER_RADIUS},
                  {'pose': np.array(pockets[3]), 'radius': self.params.POCKET_CORNER_RADIUS},
                  {'pose': np.array(pockets[4]), 'radius': self.params.POCKET_MIDDLE_RADIUS},
                  {'pose': np.array(pockets[5]), 'radius': self.params.POCKET_CORNER_RADIUS}]

  def move_cue_ball(self, angle):
    """
    Move the given joint of the given value
    :param joint: Joint to move
    :param value: Speed or torque to add to the joint
    :return:
    """
    vx=math.cos(math.radians(angle))
    vy=math.sin(math.radians(angle))
    Fx=self.params.CUE_FORCE*vx
    Fy=self.params.CUE_FORCE*vy
    self.balls[0].body.apply_impulse_at_local_point((Fx,Fy), (0,0))

  def reset(self, new_balls_pose):
    """
    Reset the world to the given arm and balls poses
    :param balls_pose:
    :param arm_position:
    :return:
    """

    ## Destroy all the circle shapes
    for shape in self.space.shapes:
        if type(shape) is pymunk.shapes.Circle:
            self.space.remove(shape)
    # Destroy all the dynamic bodies
    for body in self.space.bodies:
        if body.body_type == 0: #dynamic
            self.space.remove(body)

    # Destroy specific bodies
    #for ball in self.balls:
    #    self.space.remove(ball, ball.body)
    #    self.balls.remove(ball)

    ## Recreate the balls 
    self.create_balls(new_balls_pose)

  def step(self, dt=None):
    """
    Performs a simulator step
    :return:
    """
    if dt is None:
        self.space.step(self.dt)
    else:
       self.space.step(dt)


if __name__ == "__main__":
  
    balls_pose=[(200,103),(400,400),(500,600),(250,103)]

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
    [(1143, 96), (1122, 117), (1122, 560), (1143, 581)]
]
    phys = PhysicsSim()
    phys.create_balls(balls_pose)
    phys.create_cushions(cushions)
    phys.create_pockets(pockets)

    screen = pygame.display.set_mode((phys.params.DISPLAY_SIZE[0], phys.params.DISPLAY_SIZE[1]))
    pygame.display.set_caption('Billiard')
    clock = pygame.time.Clock()
    screen.fill((50,50,50)) 
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
    run=True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                phys.balls[0].body.apply_impulse_at_local_point((-5000,-500), (0,0))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                phys.balls[0].body.apply_impulse_at_local_point((5000,-500), (0,0))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
                phys.balls[0].body.apply_impulse_at_local_point((-5000,500), (0,0))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
                phys.balls[0].body.apply_impulse_at_local_point((5000,500), (0,0))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                phys.reset(balls_pose)
                a=1
            if event.type == pygame.QUIT:
                run = False

        screen.fill((50,50,50))
        phys.space.debug_draw(draw_options)

        #pygame.display.flip()
        pygame.display.update()

        phys.step(1/120.0)
        clock.tick(120)
