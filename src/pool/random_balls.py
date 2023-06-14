import random
import math

class RandomBalls:
    def __init__(self,ball_radius,
                computation_rectangle,
                ):
        self.ball_radius=ball_radius
        self.computation_rectangle=computation_rectangle

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        return math.hypot((x1 - x2), (y1 - y2))
    
    def generate_random_balls(self, ncirc=None):
        if ncirc is None:
            ncirc=random.randint(4, 16)# number of circles from 4 to 16
        minx=self.computation_rectangle.left_x
        maxx=self.computation_rectangle.right_x
        miny=self.computation_rectangle.top_y
        maxy=self.computation_rectangle.bottom_y

        circle_list = []
        while len(circle_list) < ncirc:
            x = random.randint(minx, maxx)
            y = random.randint(miny, maxy)
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

