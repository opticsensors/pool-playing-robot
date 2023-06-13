import pygame
import pymunk
import pymunk.pygame_util
import math

pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 678

#game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pool")

#pymunk space
space = pymunk.Space()
static_body = space.static_body
draw_options = pymunk.pygame_util.DrawOptions(screen)

#clock
clock = pygame.time.Clock()
FPS = 60

#colours
BG=(50,50,50)

#function for creating balls
def create_ball(radius, pos):
  body = pymunk.Body()
  body.position = pos
  shape = pymunk.Circle(body, radius)
  shape.mass = 5
  shape.elasticity = 0.8
  #use pivot joint to add friction
  pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0))
  pivot.max_bias = 0 # disable joint correction
  pivot.max_force = 1000 # emulate linear friction

  space.add(body, shape, pivot)
  return shape

new_ball1 = create_ball(25, (200,103))
new_ball2 = create_ball(25, (400,400))
new_ball3 = create_ball(25, (500,600))
cue_ball = create_ball(25, (250,103))
balls=[new_ball1,new_ball2,new_ball3,cue_ball]

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

#function for creating cushions
def create_cushion(poly_dims):
  body = pymunk.Body(body_type = pymunk.Body.STATIC)
  body.position = ((0, 0))
  shape = pymunk.Poly(body, poly_dims)
  shape.elasticity = 0.8
  
  space.add(body, shape)

for c in cushions:
  create_cushion(c)

#game loop
run = True
while run:

    #clock.tick(FPS)
    space.step(1 / FPS)

    #fill background
    screen.fill(BG)

    #check if any balls have been potted
    for i, ball in enumerate(balls):
      for pocket in pockets:
        ball_x_dist = abs(ball.body.position[0] - pocket[0])
        ball_y_dist = abs(ball.body.position[1] - pocket[1])
        ball_dist = math.sqrt((ball_x_dist ** 2) + (ball_y_dist ** 2))
        if ball_dist <= 50 / 2:
          #check if the potted ball was the cue ball
          #if i == len(balls) - 1:
          #  cue_ball_potted = True
          #  ball.body.position = (-100, -100)
          #  ball.body.velocity = (0.0, 0.0)
          #else:          
          print('yess')
          space.remove(ball.body)
          balls.remove(ball)
          print(space.bodies)

    #check if all the balls have stopped moving
    taking_shot = True
    for ball in balls:
      if int(ball.body.velocity[0]) != 0 or int(ball.body.velocity[1]) != 0:
        taking_shot = False

    #event handler
    for event in pygame.event.get():
        
        if event.type == pygame.MOUSEBUTTONDOWN and taking_shot == True:
            cue_ball.body.apply_impulse_at_local_point((-5000,-5000), (0,0))
        if event.type == pygame.QUIT:
            run = False

    space.debug_draw(draw_options)
    pygame.display.update()

pygame.quit()