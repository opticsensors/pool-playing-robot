import numpy as np
import cv2
import pool.utils as utils

class Pockets:
    def __init__(self,
                 pockets,
                 computation_rectangle,
                 precision):
        params=utils.Params()
        self.computation_rectangle = computation_rectangle
        self.list_of_pockets = pockets
        self.setup_pockets_todraw()
        self.setup_mouths(params.MOUTH_CORNER_WIDTH,params.MOUTH_MIDDLE_WIDTH)
        self.setup_pockets_tocompute(precision)
    
    def setup_pockets_todraw(self):
        pockets_id=np.array([1,3,4,6,2,5]).reshape(-1,1)
        arr_of_pockets = np.vstack(self.list_of_pockets)
        self.pockets_to_draw = np.hstack([pockets_id,arr_of_pockets])

    def setup_pockets_tocompute(self, precision):
        l_points_mouth=[]
        self.mouth=np.vstack((self.mouth_corners,self.mouth_middle)) #pocket 1, pocket 3, pocket 4, pocket 6, pocket 2, pocket 5
        for row in self.mouth:
            point1=row[:2]
            point2=row[2:]
            points=utils.get_equidistant_points(point1,point2,precision)
            l_points_mouth.append(points)
 
        # add ids of pockets:
        pockets_id=np.array([1,3,4,6,2,5]).reshape(-1,1)
        pockets_sub_id=np.arange(1,precision+2).reshape(-1,1)
        pocket_ids=utils.get_row_combinations_of_two_arrays(pockets_id,pockets_sub_id)
        pockets = np.vstack(l_points_mouth)
        self.pockets=np.hstack([pocket_ids,pockets])

    def get_corner_mouths(self, width_corner):
        widths_corners_horizontal=np.array([[width_corner , 0],
                                            [-width_corner, 0],
                                            [-width_corner, 0],
                                            [width_corner , 0]])
        widths_corners_vertical=np.array([[0, width_corner ],
                                          [0, width_corner],
                                          [0, -width_corner],
                                          [0, -width_corner ]])
        horizontal_points = self.computation_rectangle.rectangle + widths_corners_horizontal
        vertical_points = self.computation_rectangle.rectangle + widths_corners_vertical
        return np.hstack((horizontal_points,vertical_points))
        
    def get_middle_mouths(self, width_middle):

        widths_middle_top=np.array([[-width_middle , 0],
                                    [-width_middle, 0]])
        widths_middle_bottom=np.array([[width_middle , 0],
                                       [width_middle, 0]])     

        middle_top_points= self.middle_points + widths_middle_top
        middle_bottom_points=self.middle_points + widths_middle_bottom
        return np.hstack((middle_top_points,middle_bottom_points))

    def setup_mouths(self, width_corner, width_middle):
        middle_x = (self.list_of_pockets[4][0] + self.list_of_pockets[5][0]) / 2
        top_y = self.computation_rectangle.top_y
        bottom_y = self.computation_rectangle.bottom_y
        middle_top_point = np.array([middle_x, top_y])
        middle_bottom_point = np.array([middle_x, bottom_y])
        self.middle_points = np.vstack((middle_top_point, middle_bottom_point))    
        self.mouth_corners=self.get_corner_mouths(width_corner)
        self.mouth_middle=self.get_middle_mouths(width_middle)

class Cushions:
    def __init__(self,
                 computation_rectangle,
                 cushion_ranges,
                 ):
        self.computation_rectangle=computation_rectangle
        self.cushion_ranges=cushion_ranges
        self.setup_cushions()

    def setup_cushions(self):
        x_range1, x_range2, y_range = self.cushion_ranges
        self.cushions = np.array([[x_range1[0], self.computation_rectangle.top_y, x_range1[1], self.computation_rectangle.top_y ],
                         [x_range2[0], self.computation_rectangle.top_y, x_range2[1], self.computation_rectangle.top_y],
                         [x_range1[0], self.computation_rectangle.bottom_y, x_range1[1], self.computation_rectangle.bottom_y ],
                         [x_range2[0], self.computation_rectangle.bottom_y, x_range2[1], self.computation_rectangle.bottom_y],
                         [self.computation_rectangle.left_x, y_range[0], self.computation_rectangle.left_x, y_range[1]],
                         [self.computation_rectangle.right_x, y_range[0], self.computation_rectangle.right_x, y_range[1]],
                         ])

class PoolFrame:
    def __init__(self,
                 ball_radius = None,
                 computation_rectangle = None,
                 cushion_ranges = None,
                 pockets_positions = None,
                 precision = 0):
        
        params=utils.Params()
        if ball_radius is None:
            self.ball_radius=params.BALL_RADIUS
        else:
            self.ball_radius=ball_radius

        if computation_rectangle is None:
            self.computation_rectangle=params.COMPUTATIONAL_RECTANGLE
        else:
            self.computation_rectangle=computation_rectangle
        
        if cushion_ranges is None:
            self.cushion_ranges=params.CUSHION_RANGES
        else:
            self.cushion_ranges=cushion_ranges

        if pockets_positions is None:
            self.pockets_positions=params.POCKETS
        else:
            self.pockets_positions=pockets_positions

        self.rectangle = params.COMPUTATIONAL_RECTANGLE
        self.pockets = Pockets(pockets = self.pockets_positions,
                               computation_rectangle = self.computation_rectangle,
                               precision = precision)
        self.cushions = Cushions(computation_rectangle = self.rectangle,
                                 cushion_ranges = self.cushion_ranges)
   
        self.top_y = self.computation_rectangle.top_y
        self.right_x = self.computation_rectangle.right_x
        self.bottom_y = self.computation_rectangle.bottom_y
        self.left_x = self.computation_rectangle.left_x
        self.top_left = self.computation_rectangle.top_left
        self.top_right = self.computation_rectangle.top_right
        self.bottom_right = self.computation_rectangle.bottom_right
        self.bottom_left = self.computation_rectangle.bottom_left
        x_range1, x_range2, y_range = self.cushions.cushion_ranges
        self.xrange_horizontal_left_cushion = x_range1
        self.xrange_horizontal_right_cushion = x_range2
        self.xrange_vertical_cushion = y_range

    @staticmethod
    def draw_pocket(img,point,radius):
        cv2.circle(img, point, radius, (0, 255, 255), 8)
        x,y=point
        cv2.line(img, (x+radius, y), (x-radius, y), (0, 255, 255), thickness=8)
        cv2.line(img, (x, y-radius), (x, y+radius), (0, 255, 255), thickness=8)
        return img
    
    @staticmethod
    def draw_segment(img, cushion):
        point1=cushion[:2].astype(np.int32)
        point2=cushion[2:].astype(np.int32)
        cv2.line(img, point1, point2, (173, 203, 248), thickness=20)
        return img
    
    def draw_frame(self, img):
        params=utils.Params()
        cv2.rectangle(img, self.top_left, 
                      self.bottom_right, 
                      (255,255,255), 
                      3)
        for row in self.pockets.pockets_to_draw:
            pocket_id=row[0]
            point=row[1:]
            if pocket_id in [2,5]: #middle pockets are smaller
                img=self.draw_pocket(img, point, params.POCKET_MIDDLE_RADIUS) 
            else: #[1,3,4,6]
                img=self.draw_pocket(img, point, params.POCKET_CORNER_RADIUS) 

        for row in self.pockets.mouth:
            img=self.draw_segment(img,row)

        for row in self.cushions.cushions:
            img=self.draw_segment(img,row)
        
        return img

    def draw_pool_balls(self, img, d_centroids):
        for ball_num in d_centroids:
            x,y=d_centroids[ball_num]
            img=cv2.putText(img, "#{}".format(int(ball_num)), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            img=cv2.circle(img, (int(x), int(y)), 8, (49, 125, 237), -1)
            img=cv2.circle(img, (int(x), int(y)), self.ball_radius, (49, 125, 237), 8)

        return img
