import random
import math
from pool.utils import Params

class RandomBalls:
    def __init__(self,
                 ball_radius=None,
                 computation_rectangle=None,
                ):
        params=Params()
        if ball_radius is None:
            self.ball_radius = params.BALL_RADIUS
        else:
            self.ball_radius=ball_radius

        if computation_rectangle is None:
            self.computation_rectangle=params.COMPUTATIONAL_RECTANGLE
        else:
            self.computation_rectangle=computation_rectangle
            
        self.minx=self.computation_rectangle.left_x
        self.maxx=self.computation_rectangle.right_x
        self.miny=self.computation_rectangle.top_y
        self.maxy=self.computation_rectangle.bottom_y

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        return math.hypot((x1 - x2), (y1 - y2))
    
    def generate_random_balls(self, ncirc=None):
        if ncirc is None:
            ncirc=random.randint(4, 16)# number of circles from 4 to 16

        circle_list = []
        while len(circle_list) < ncirc:
            x = random.randint(self.minx, self.maxx)
            y = random.randint(self.miny, self.maxy)
            if not any((x2, y2) for x2, y2 in circle_list if self.euclidean_distance(x, y, x2, y2) < 2*self.ball_radius):
                circle_list.append((x, y))  

        cue_and_8ball=[0,8]
        npot=ncirc-2 #goes from 2 to 14
        # at max there are 7 solid balls and 7 strip balls 
        if npot-7>0:
            min_num_solid_balls=npot-7
        else:
            min_num_solid_balls=1
        nsolid=random.randint(min_num_solid_balls, npot-min_num_solid_balls)
        nstrip=npot-nsolid
        solid_balls=random.sample(range(1,8), nsolid)
        strip_balls=random.sample(range(9,16), nstrip)
        balls=cue_and_8ball+solid_balls+strip_balls
        return dict(zip(balls, circle_list))

    def generate_random_positions_given_balls(self, ball_numbers):

        assert 0 in ball_numbers and 8 in ball_numbers

        circle_list = []
        ncirc = len(ball_numbers)
        while len(circle_list) < ncirc:
            x = random.randint(self.minx, self.maxx)
            y = random.randint(self.miny, self.maxy)
            if not any((x2, y2) for x2, y2 in circle_list if self.euclidean_distance(x, y, x2, y2) < 2*self.ball_radius):
                circle_list.append((x, y))  

        return dict(zip(ball_numbers, circle_list))