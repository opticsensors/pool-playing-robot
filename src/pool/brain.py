import cv2
import numpy as np
from matplotlib import pyplot as plt

class Brain(object):
    """
    A class that finds the optimal shot 

    Attributes
    ----------    
    ...

    """
    

    RECTANGLE_AREA=382352 #HxV
    BALL_AREA=1134.115 #PI*RADI^2
    RATIO_BALL_RECTANGLE=0.00296615132

    def __init__(self,
                d_centroids={}
        ):
        
        self.d_centroids=d_centroids

    @staticmethod
    def _draw_pocket(img,point,radius):
        cv2.circle(img, point, radius, (0, 255, 255), 8)
        x,y=point
        cv2.line(img, (x+radius, y), (x-radius, y), (0, 255, 255), thickness=8)
        cv2.line(img, (x, y-radius), (x, y+radius), (0, 255, 255), thickness=8)
        return img
    
    def pool_frame(self, 
                   img, 
                   x_pocket_top_left,
                   y_pocket_top_left,
                   x_pocket_top_middle,
                   y_pocket_top_middle,
                   horizontal_left_offset,
                   vertical_top_offset,
                   width_corner,
                   width_middle,
                   **kwargs):
        
        H=img.shape[0]
        W=img.shape[1]
        
        param = {  # with defaults
            'x_pocket_top_left': x_pocket_top_left,
            'y_pocket_top_left': y_pocket_top_left,
            'x_pocket_top_right': W-x_pocket_top_left,
            'y_pocket_top_right': y_pocket_top_left,
            'x_pocket_bottom_left': W-x_pocket_top_left,
            'y_pocket_bottom_left': H-y_pocket_top_left,
            'x_pocket_bottom_right': x_pocket_top_left,
            'y_pocket_bottom_right': H-y_pocket_top_left,
            'x_pocket_top_middle': x_pocket_top_middle,
            'y_pocket_top_middle': y_pocket_top_middle,
            'x_pocket_bottom_middle':x_pocket_top_middle,
            'y_pocket_bottom_middle':H-y_pocket_top_middle,
            'radius_pocket_corners' : 210,
            'radius_pocket_middle' : 112, 
            'horizontal_left_offset': horizontal_left_offset,
            'horizontal_right_offset': horizontal_left_offset,
            'vertical_top_offset': vertical_top_offset,
            'vertical_bottom_offset': vertical_top_offset                                    
            }
        param.update(kwargs)
        
        self.pocket_top_left = (param['x_pocket_top_left'], param['y_pocket_top_left'])
        self.pocket_top_right = (param['x_pocket_top_right'], param['y_pocket_top_right'])
        self.pocket_bottom_left = (param['x_pocket_bottom_left'], param['y_pocket_bottom_left'])
        self.pocket_bottom_right = (param['x_pocket_bottom_right'], param['y_pocket_bottom_right'])
        self.pocket_top_middle = (param['x_pocket_top_middle'], param['y_pocket_top_middle'])
        self.pocket_bottom_middle = (param['x_pocket_bottom_middle'], param['y_pocket_bottom_middle'])
        
        self._draw_pocket(img, self.pocket_top_left, param['radius_pocket_corners'])
        self._draw_pocket(img, self.pocket_top_right, param['radius_pocket_corners'])
        self._draw_pocket(img, self.pocket_bottom_left, param['radius_pocket_corners'])
        self._draw_pocket(img, self.pocket_bottom_right, param['radius_pocket_corners'])
        self._draw_pocket(img, self.pocket_top_middle, param['radius_pocket_middle'])
        self._draw_pocket(img, self.pocket_bottom_middle, param['radius_pocket_middle'])

        self.mouth_top_left1 = (param['horizontal_left_offset'], param['vertical_top_offset']+width_corner)
        self.mouth_top_left2 = (param['horizontal_left_offset'] + width_corner, param['vertical_top_offset'])

        self.mouth_top_right3 = (W-param['horizontal_right_offset']-width_corner, param['vertical_top_offset'])
        self.mouth_top_right4 = (W-param['horizontal_right_offset'], param['vertical_top_offset']+width_corner)

        self.mouth_bottom_right5 = (W-param['horizontal_right_offset'], H-param['vertical_bottom_offset']-width_corner)
        self.mouth_bottom_right6 = (W-param['horizontal_right_offset']-width_corner, H-param['vertical_bottom_offset'])

        self.mouth_bottom_left7 = (param['horizontal_left_offset']+width_corner, H-param['vertical_bottom_offset'])
        self.mouth_bottom_left8 = (param['horizontal_left_offset'], H-param['vertical_bottom_offset']-width_corner)

        self.mouth_top_middle1 = (x_pocket_top_middle-width_middle, param['vertical_top_offset'])
        self.mouth_top_middle2 = (x_pocket_top_middle+width_middle, param['vertical_top_offset'])

        self.mouth_bottom_middle3 = (x_pocket_top_middle-width_middle, H-param['vertical_bottom_offset'])
        self.mouth_bottom_middle4 = (x_pocket_top_middle+width_middle, H-param['vertical_bottom_offset'])

        cv2.line(img, self.mouth_top_left2, self.mouth_top_right3, (0, 255, 0), thickness=3)
        cv2.line(img, self.mouth_top_right4, self.mouth_bottom_right5, (0, 255, 0), thickness=3)
        cv2.line(img, self.mouth_bottom_right6, self.mouth_bottom_left7, (0, 255, 0), thickness=3)
        cv2.line(img, self.mouth_bottom_left8, self.mouth_top_left1, (0, 255, 0), thickness=3)

        cv2.line(img, self.mouth_top_left1, self.mouth_top_left2, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_top_right3, self.mouth_top_right4, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_bottom_right5, self.mouth_bottom_right6, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_bottom_left7, self.mouth_bottom_left8, (255, 255, 0), thickness=20)

        cv2.line(img, self.mouth_top_middle1, self.mouth_top_middle2, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_bottom_middle3, self.mouth_bottom_middle4, (255, 255, 0), thickness=20)

        return img
