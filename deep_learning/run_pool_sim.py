import cv2
from pool.pool_frame import Rectangle
from pool.random_balls import RandomBalls
from pool.pool_sim import PoolSimulation

img = cv2.imread(f'./table.png')
H=img.shape[0]
W=img.shape[1]

img_rectangle=Rectangle(top_left=(0,0),
                        bottom_right=(W,H))
computation_rectangle=img_rectangle.get_rectangle_with_offsets((127, 127, 127, 127))

random_balls=RandomBalls(ball_radius=25,
                         computation_rectangle=computation_rectangle)
d_centroids=random_balls.generate_random_balls()


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

sim=PoolSimulation(W,H,FPS=120,ball_radius=25, cushions=cushions, pockets=pockets)
sim.create_balls(d_centroids)
sim.apply_impulse(135, 15000)
#sim.reset(d_centroids)
sim.run()