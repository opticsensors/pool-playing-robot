import numpy as np
import os

class Params(object):
  """
  Define all python like used parameters
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
    self.ARUCOS_POOL_FRAME = [[2,3],  #top                 
                              [4,5],  #right            
                              [6,7],  #bottom         
                              [0,1]]  #left 
    
    ## CLASSIC CV PARAMS
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
        'white'   : [227, 127, 163],
        'yellow'  : [216, 132, 189],
        'blue'    : [89, 134, 103],
        'red'     : [147, 181, 181],
        'purple'  : [78, 137, 119],
        'orange'  : [180, 165 , 175],
        'green'   : [110, 106, 132],
        'burgundy': [121, 154, 154],
        'black'   : [58, 128, 130]
    }
    
    self.WHITE_LOWER_LAB=[175, 0, 0]
    self.WHITE_UPPER_LAB=[255, 147, 164]
    self.WHITE_LOWER_HSV=[0, 0, 120]
    self.WHITE_UPPER_HSV=[56, 147, 255]
    
    self.RECTANGLE_AREA=self.DISPLAY_SIZE[0]*self.DISPLAY_SIZE[1] # W*H
    self.BALL_AREA=np.pi*self.BALL_RADIUS**2 #PI*RADI^2
    self.RATIO_BALL_RECTANGLE = self.BALL_AREA/self.RECTANGLE_AREA

    ## ERROR ANALYSIS PARAMS
    # metrics in cm
    self.BALL_RADIUS_MM=38/2
    self.DISPLAY_SIZE_MM=(924,524)
    self.POCKETS_MM=[[38.447187,38.39291465],
                     [462.2112483,38.39291465],
                     [885.9753086,38.39291465],
                     [885.9753086,485.60708],
                     [462.2112483,485.60708],
                     [38.447187,485.60708]]
    
    ## PIXEL TO STEP CALIBRATION PARAMS
    self.CM_TO_STEPS = 0.00794
    self.GRID_SIZE_CM = (70.25, 38.5) # WxH

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

def angle_between_3_points( a,b,c):
    ba = a - b
    bc = c - b
    return angle_between_two_vectors(ba,bc)

def line_intersect(a1, a2, b1, b2): # TODO change name to line_intersect_given_points
    T = np.array([[0, -1], [1, 0]])
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1


def intersection_circle_line(slope,intercept, r,center):
    """
    Computes the points of intersection (if any) between a line 
    and a circumference given a slope, an intercection, a radii 
    and a center.
    """
    #intersection point between the above line and a cercle of center T and radii 2r
    new_intercept=intercept+slope*center[:,0]-center[:,1]
    _A=1+slope**2
    _B=2*slope*new_intercept
    _C=new_intercept**2-(2*r)**2

    x1=(-_B+np.sqrt(_B**2-4*_A*_C))/(2*_A) + center[:,0]
    x2=(-_B-np.sqrt(_B**2-4*_A*_C))/(2*_A) + center[:,0]
    y1=slope*x1+intercept
    y2=slope*x2+intercept

    #we need to choose which one is the correct intersection point
    X_calculated1=np.c_[x1,y1]
    X_calculated2=np.c_[x2,y2]

    return X_calculated1,X_calculated2

def intersection_two_circles(P0,P1,r0,r1):
    """
    Computes the points of intersection (if any) between two
    circles of radiis r0,r1 and centers P0,P1
    """

    # distance from P0 to P1  
    P0P1 = P1-P0  
    d=np.sqrt((P0P1[:,0])**2+(P0P1[:,1])**2)
    # distance from P0 to P2 (being P2 the point of intersection between lines P0P1 and X1X2)
    # (X1,X2 are the intersection points that we want to find)
    a=(d**2+r0**2-r1**2)/(2*d)
    # distance from P1 to P2
    b=d-a
    # distance from P2 to X1 = distance from P2 to X2
    h=np.sqrt(r0**2-a**2)
    # convert points to numpy array

    auxiliar_point=(P0.T * (b/d)).T+(P1.T * (a/d)).T

    intersec1_x=auxiliar_point[:,0]+(h/d)*P0P1[:,1]
    intersec2_x=auxiliar_point[:,0]-(h/d)*P0P1[:,1]

    intersec1_y=auxiliar_point[:,1]-(h/d)*P0P1[:,0]
    intersec2_y=auxiliar_point[:,1]+(h/d)*P0P1[:,0]

    X1=np.column_stack((intersec1_x,intersec1_y))
    X2=np.column_stack((intersec2_x,intersec2_y))

    return X1, X2

def generate_random_number_inside_circle(center,R):
    """
    generates random numbers inside a circle of radii=R and center=center
    """
    theta = np.random.uniform(0,2*np.pi, center.shape[0])
    radius = np.random.uniform(0,R, center.shape[0]) ** 0.5
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    x=center[:,0]+x
    y=center[:,1]+y

    return np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)

def generate_random_numbers_inside_rectangle(num_points,size,safety_distance):
    """
    generates random numbers inside a rectangle of left bottom vertex (0,0)
    and right top vertex (W,H). We also consider a safety distance from 
    the sides of the rectangle.
    """
    W,H=size
    xlist = np.random.uniform(0+safety_distance, W-safety_distance, num_points)
    ylist = np.random.uniform(0+safety_distance, H-safety_distance, num_points)
    real_points=np.concatenate((xlist.reshape(-1,1),ylist.reshape(-1,1)), axis=1)
    return real_points