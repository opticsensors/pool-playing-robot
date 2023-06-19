import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np
from pool.utils import Params

class PhysicsSim(object):
  """
  Physics simulator
  """
  def __init__(self, params=None):
    """
    Constructor
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
    self.total_collisions = 0
    self.ball_collision_happened = False
    self.first_ball_collision = None

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
        shape.collision_type = 4
        self.space.add(body, shape)
        self.cushions.append(c)

  def create_balls(self, balls_pose):
    """
    Creates the balls in the simulation at the given positions
    :param balls_pose: Initial pose of the ball in table RF
    :return:
    """
    ## dict of balls in simulation
    self.balls = {}

    for idx, pose in balls_pose.items():
        mass = self.params.BALL_MASS
        radius = self.params.BALL_RADIUS
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        body = pymunk.Body(mass, inertia)
    
        #body = pymunk.Body()
        body.position = pose
        shape = pymunk.Circle(body, self.params.BALL_RADIUS)
        #shape.mass = self.params.BALL_MASS
        shape.elasticity = self.params.BALL_ELASTICITY
        if idx in [1,2,3,4,5,6,7]:
          shape.collision_type = 0
        elif idx in [9,10,11,12,13,14,15]:
          shape.collision_type = 1
        elif idx == 0:
           shape.collision_type = 2
        elif idx == 8:
           shape.collision_type = 3
        #use pivot joint to add friction
        pivot = pymunk.PivotJoint(self.static_body, body, (0, 0), (0, 0))
        pivot.max_bias = 0 # disable joint correction
        pivot.max_force = self.params.BALL_FRICTION # emulate linear friction
        self.space.add(body, shape, pivot)
        self.balls[idx] = shape

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
    Move the cue ball the given value
    :return:
    """
    vx=math.cos(math.radians(angle))
    vy=math.sin(math.radians(angle))
    Fx=self.params.CUE_FORCE*vx
    Fy=self.params.CUE_FORCE*vy
    self.balls[0].body.apply_impulse_at_local_point((Fx,Fy), (0,0))

  def callback_function_default(self,arbiter, space, data):
    self.total_collisions += 1
    return True
  
  def callback_function_cue_solid(self,arbiter, space, data):
    self.total_collisions += 1
    if not self.ball_collision_happened:
      self.first_ball_collision = 0
      self.ball_collision_happened = True
    return True
  
  def callback_function_cue_strip(self,arbiter, space, data):
    self.total_collisions += 1
    if not self.ball_collision_happened:
      self.first_ball_collision = 1
      self.ball_collision_happened = True
    return True
  
  def handle_collisions(self):
    handler_default = self.space.add_default_collision_handler()
    handler20 = self.space.add_collision_handler(2,0)
    handler21 = self.space.add_collision_handler(2,1)
    handler_default.begin = self.callback_function_default
    handler20.begin = self.callback_function_cue_solid
    handler21.begin = self.callback_function_cue_strip

  def reset(self, new_balls_pose):
    """
    Reset the world to the given balls poses
    :param balls_pose:
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
    self.total_collisions = 0
    self.ball_collision_happened = False
    self.first_ball_collision = None

  def step(self, dt=None):
    """
    Performs a simulator step
    :return:
    """
    if dt is None:
        self.space.step(self.dt)
    else:
       self.space.step(dt)
    self.collision_occurs = 0


if __name__ == "__main__":
  
    balls_pose={0:(200,103),8:(400,400),9:(500,500),1:(250,103)}

    #create six pockets on table
    pockets = [
    (55, 63),
    (592, 48),
    (1134, 64),
    (1134, 616),
    (592, 629),
    (55, 616),
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
    phys.handle_collisions()

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
                phys.move_cue_ball(225)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                phys.move_cue_ball(315)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
                phys.move_cue_ball(135)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
                phys.move_cue_ball(45)
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

        #debug
        print(phys.total_collisions,phys.ball_collision_happened,phys.first_ball_collision)


