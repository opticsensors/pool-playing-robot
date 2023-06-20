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

    ## System params 
    # pool frame params 
    self.DISPLAY_SIZE = (1200, 678) # W x H 
    self.POCKET_CORNER_RADIUS = 25
    self.POCKET_MIDDLE_RADIUS = 25
    self.MOUTH_CORNER_WIDTH = 80
    self.MOUTH_MIDDLE_WIDTH = 70
    self.DISPLAY_RECTANGLE = Rectangle(top_left=(0,0), bottom_right=self.DISPLAY_SIZE)
    self.COMPUTATIONAL_RECTANGLE = self.DISPLAY_RECTANGLE.get_rectangle_with_offsets((260, 250, 240, 250))

    self.POCKETS = [(55, 63),     # top left          
                    (1134, 64),   # top right       
                    (1134, 616),  # bottom right     
                    (55, 616),    # bottom left
                    (592, 48),    # top middle    
                    (592, 629)]   # bottom middle    

    self.CUSHIONS = [[(88, 56), (109, 77), (555, 77), (564, 56)],        #order doesn't matter here     
                    [(621, 56), (630, 77), (1081, 77), (1102, 56)],         
                    [(89, 621), (110, 600),(556, 600), (564, 621)],         
                    [(622, 621), (630, 600), (1081, 600), (1102, 621)],         
                    [(56, 96), (77, 117), (77, 560), (56, 581)],              
                    [(1143, 96), (1122, 117), (1122, 560), (1143, 581)]]         
            
    self.CUSHION_RANGES = ((445,2154),    # top and bottom left cushion ranges
                           (2729,4338),   # top and bottom right cushion ranges
                           (476,2223))    # left and right cushion ranges

    #physics params
    self.BALL_RADIUS = 25
    self.BALL_ELASTICITY = 0.8
    self.BALL_FRICTION = 1000
    self.BALL_MASS = 5
    self.BALL_TERMINAL_VELOCITY = 0

    self.CUSHION_ELASTICITY = 0.8
    self.CUE_FORCE = 6000

    # Graphic params
    self.TARGET_FPS = 60
    self.TIME_STEP = 1.0 / self.TARGET_FPS
    self.MAX_ENV_STEPS = 10


class Rectangle:
    def __init__(self, top_left, bottom_right):

        self.rectangle = np.array([[top_left[0],     top_left[1]],       # top left 
                                   [bottom_right[0], top_left[1]],   # top right      
                                   [bottom_right[0], bottom_right[1]],   # bottom right  
                                   [top_left[0],     bottom_right[1]]])      # bottom left   
    @property
    def width(self):
        return self.rectangle[2,0]-self.rectangle[0,0]
    @property
    def height(self):
        return self.rectangle[2,1]-self.rectangle[0,1]
    @property
    def top_y(self):
        return self.rectangle[0,1]
    @property
    def bottom_y(self):
        return self.rectangle[2,1]
    @property
    def left_x(self):
        return self.rectangle[0,0]
    @property
    def right_x(self):
        return self.rectangle[2,0]
    @property
    def top_left(self):
        return self.rectangle[0,:]
    @property
    def top_right(self):
        return self.rectangle[1,:]
    @property
    def bottom_right(self):
        return self.rectangle[2,:]
    @property
    def bottom_left(self):
        return self.rectangle[3,:]

    def get_rectangle_with_offsets(self, different_offsets):
        offset_top_y, offset_right_x, offset_bottom_y, offset_left_x = different_offsets
        top_left=self.top_left+np.array([offset_left_x , offset_top_y])
        bottom_right=self.bottom_right+np.array([-offset_right_x, -offset_bottom_y])
        return Rectangle(top_left, bottom_right)
    
    def set_rectangle_with_offsets(self, different_offsets):
        offset_top_y, offset_right_x, offset_bottom_y, offset_left_x = different_offsets
        offsets_matrix = np.array([[offset_left_x , offset_top_y],
                                   [-offset_right_x, offset_top_y],
                                   [-offset_right_x, -offset_bottom_y],
                                   [offset_left_x , -offset_bottom_y],])
        self.rectangle = self.rectangle+offsets_matrix


def get_file_from_data():
    raise NotImplementedError

def get_row_combinations_of_two_arrays(array1,array2):
    if len(array1.shape)==1:
        array1=array1.reshape(1,2)

    if len(array2.shape)==1:
        array2=array2.reshape(1,2)

    a = np.repeat(array1, array2.shape[0], axis=0)
    b = np.tile(array2, (array1.shape[0],1))
    result = np.hstack([a,b])

    return result

def get_equidistant_points(p1, p2, parts):
    if parts==0:
        points=(p1+p2)/2
    else:
        points_separated=np.linspace(p1[0], p2[0], parts+1),np.linspace(p1[1], p2[1], parts+1)
        points=np.column_stack((points_separated[0],points_separated[1]))
    return points

def angle_between_two_vectors(u,v):
    dot = u[:,0]*v[:,0] + u[:,1]*v[:,1] # equivalent to np.sum(u*v, axis=1)
    cosine_angle = dot / (np.linalg.norm(u, axis=1)* np.linalg.norm(v, axis=1))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def line_intersect(a1, a2, b1, b2):
    T = np.array([[0, -1], [1, 0]])
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1
