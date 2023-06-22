import numpy as np
import os

class Params(object):
  """
  Define all used parameters
  """
  def __init__(self):
    """
    Constructor
    """
    ## REPO PATH
    self.PATH_REPO = os.path.abspath(
        os.path.join(
            os.path.abspath(__file__),
            "..","..","..",
        )
    )


    ## POOL FRAME PARAMS
    self.DISPLAY_SIZE = (4822, 2719) # W x H 
    self.POCKET_CORNER_RADIUS = 210
    self.POCKET_MIDDLE_RADIUS = 112
    self.MOUTH_CORNER_WIDTH = 80
    self.MOUTH_MIDDLE_WIDTH = 70
    self.DISPLAY_RECTANGLE = Rectangle(top_left=(0,0), bottom_right=self.DISPLAY_SIZE)
    self.COMPUTATIONAL_RECTANGLE = self.DISPLAY_RECTANGLE.get_rectangle_with_offsets((260, 250, 240, 250))

    self.POCKETS = [(96, 120),                                            # top left          
                    (self.DISPLAY_SIZE[0]-96, 120),                       # top right       
                    (self.DISPLAY_SIZE[0]-96, self.DISPLAY_SIZE[1]-120),  # bottom right     
                    (96, self.DISPLAY_SIZE[1]-120),                       # bottom left
                    (2430, 164),                                          # top middle    
                    (2430, 2574)]                                         # bottom middle    

    self.CUSHIONS = [[(308, 0), (2304, 0), (2200, 138), (419, 138)],        #order doesn't matter here     
                    [(2552, 0), (4516, 0), (4382, 138), (2660, 138)],         
                    [(4822, 269), (4822, 2452),(4665, 2288), (4665, 413)],         
                    [(4515, 2719), (2552, 2719), (2660, 2571), (4382, 2571)],         
                    [(2305, 2719), (308, 2719), (419, 2571), (2201, 2571)],              
                    [(0, 2452), (0, 269), (154, 414), (154, 2287)]]         
            
    self.CUSHION_RANGES = ((445,2154),    # top and bottom left cushion ranges
                           (2729,4338),   # top and bottom right cushion ranges
                           (476,2223))    # left and right cushion ranges
      
    #PHYSICS PARAMS
    self.BALL_RADIUS = 102
    self.BALL_ELASTICITY = 0.8
    self.BALL_FRICTION = 20000
    self.BALL_MASS = 5
    self.BALL_TERMINAL_VELOCITY = 400
    self.CUSHION_ELASTICITY = 0.6
    self.CUE_FORCE = 50000

    # GRAPHIC PARAMS
    self.TARGET_FPS = 120
    self.TIME_STEP = 1.0 / self.TARGET_FPS
    self.MAX_ENV_STEPS = 10

    ## EYE PARAMS
    #we map each color with a number:
    self.NUM_TO_COLOR={
        0:'white' ,
        1:'yellow',
        2:'blue',
        3:'red',
        4:'purple',
        5:'orange',
        6:'green',
        7:'burgundy',
        8:'black'
    }

    #decision tree: knowing ball color and type, we can know its number
    self.COLOR_AND_TYPE_TO_NUM={
        'white'   : {'cue ball': 0},
        'yellow'  : {'solid':1, 'striped':9},
        'blue'    : {'solid':2, 'striped':10},
        'red'     : {'solid':3, 'striped':11},
        'purple'  : {'solid':4, 'striped':12},
        'orange'  : {'solid':5, 'striped':13},
        'green'   : {'solid':6, 'striped':14},
        'burgundy': {'solid':7, 'striped':15},
        'black'   : {'solid':8}
    }
    
    #lab calibration results of ball colors
    #sorted by rows like: white, yellow, blue, red,... (same order as pool balls)
    self.COLOR_TO_LAB={
        'white'   : [227, 130, 148],
        'yellow'  : [216.18407452, 124.80355131, 195.73519079],
        'blue'    : [89.64986175, 134.93342045, 103.84520826],
        'red'     : [147.45829885, 172.49603901, 178.22393935],
        'purple'  : [78.6311673, 135.5586135, 123.057679],
        'orange'  : [180.62554741, 151.8795586 , 187.98600759],
        'green'   : [110.05696026, 106.50088757, 132.41375192],
        'burgundy': [121.56706638, 160.19773476, 158.41379201],
        'black'   : [58, 129, 136]
    }
    
    self.WHITE_LOWER_LAB=[175, 0, 0]
    self.WHITE_UPPER_LAB=[255, 147, 164]
    self.WHITE_LOWER_HSV=[0, 0, 179]
    self.WHITE_UPPER_HSV=[180, 106, 255]
    
    self.RECTANGLE_AREA=self.DISPLAY_SIZE[0]*self.DISPLAY_SIZE[1] # W*H
    self.BALL_AREA=np.pi*self.BALL_RADIUS**2 #PI*RADI^2
    self.RATIO_BALL_RECTANGLE = self.BALL_AREA/self.RECTANGLE_AREA

    self.ARUCOS_POOL_FRAME = [[2,3],  #top                 
                              [4,5],  #right            
                              [6,7],  #bottom         
                              [0,1]]  #left 

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
    return angle #in radians

def line_intersect(a1, a2, b1, b2):
    T = np.array([[0, -1], [1, 0]])
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1
