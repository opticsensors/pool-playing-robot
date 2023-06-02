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

    BALL_RADIUS=102 

    NUM_TO_COLOR={
        0:'white' ,
        1:'yellow',
        2:'blue',
        3:'red',
        4:'purple',
        5:'orange',
        6:'green',
        7:'burgundy',
        8:'black',
        9:'yellow',
        10:'blue',
        11:'red',
        12:'purple',
        13:'orange',
        14:'green',
        15:'burgundy',
    }

    def __init__(self,
                d_centroids={},
                ball_radius=None
        ):
        
        self.d_centroids=d_centroids
        if ball_radius is None:
            self.ball_radius=Brain.BALL_RADIUS
        else:
            self.ball_radius=ball_radius

    @staticmethod
    def _draw_pocket(img,point,radius):
        cv2.circle(img, point, radius, (0, 255, 255), 8)
        x,y=point
        cv2.line(img, (x+radius, y), (x-radius, y), (0, 255, 255), thickness=8)
        cv2.line(img, (x, y-radius), (x, y+radius), (0, 255, 255), thickness=8)
        return img

    @staticmethod
    def get_equidistant_points(p1, p2, parts):
        points_separated=np.linspace(p1[0], p2[0], parts+1),np.linspace(p1[1], p2[1], parts+1)
        points=np.column_stack((points_separated[0],points_separated[1]))
        return points
    
    @staticmethod
    def angle_abc(a,b,c):
        """
        Computes the angle between 3 points 
        (point b is the vertex)

        Parameters
        ----------    
            a,b,c: numpy array of shape (2,)
                x, y coordinates of the point
        Returns
        -------
            angle: numpy float64

        """
        ba = a - b
        bc = c - b
        dot = ba[:,0]*bc[:,0] + ba[:,1]*bc[:,1]
        cosine_angle = dot / (np.linalg.norm(ba, axis=1)* np.linalg.norm(bc, axis=1))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def get_balls_to_be_pocket(self,ball_type):
        
        l_detected=[key for key in self.d_centroids]
        if ball_type=='solid':
            l_type=[1,2,3,4,5,6,7]
        elif ball_type=='strip':
            l_type=[9,10,11,12,13,14,15]

        l_detected_type=list(set(l_type) & set(l_detected))
        dct_detected_type = {key: self.d_centroids[key] for key in l_detected_type}
        arr_detected_type= np.array([list(val) for val in dct_detected_type.values()])
        arr_ids_detected_type=np.array(l_detected_type).reshape(-1,1)
        return np.hstack((arr_ids_detected_type,arr_detected_type))

    def get_cue_and_8ball(self):
        if 0 in self.d_centroids:
            arr_cue = np.array(self.d_centroids[0])
        else:   
            raise ValueError('cue ball not detected, pool shot cannot be executed')
        if 8 in self.d_centroids:
            arr_8ball = np.array(self.d_centroids[8])
        else:   
            raise ValueError('8 ball not detected, game over')

        return arr_cue, arr_8ball
    
    def get_other_balls(self, T):

        l_detected=[key for key in self.d_centroids]
        l_detected.remove(0)
        dct_detected = {key: self.d_centroids[key] for key in l_detected}
        arr_detected = np.array([list(val) for val in dct_detected.values()])
        arr_ids_detected = np.array(l_detected).reshape(-1,1)
        all_detected_balls_except_cue=np.hstack((arr_ids_detected,arr_detected))

        T_and_others = np.array([]).reshape(0,6)
        for row in T:
            target_id=row[0]
            #other_balls = np.delete(all_detected_balls_except_cue, (target_id), axis=0)
            other_balls = all_detected_balls_except_cue[all_detected_balls_except_cue[:,0]!=target_id]
            row=np.repeat(row.reshape(1,-1),other_balls.shape[0],0)
            row_and_others=np.hstack((row,other_balls))
            T_and_others=np.vstack((T_and_others,row_and_others))
        return T_and_others

    def get_point_combinations(self,C,T,P):
        if len(C.shape)==1:
            C=C.reshape(1,2)
        repeated_P=np.tile(P,(len(T),1))
        repeated_T=np.repeat(T,len(P),0)
        repeated_C=np.repeat(C,repeated_P.shape[0],axis=0)
        return repeated_C,repeated_T,repeated_P

    def find_X1_and_X2(self,C,T):

        # distance from C to T    
        d=np.linalg.norm(T-C, axis=1)
        # distance from C to C2 (being C2 the point of intersection between lines CT and X1X2)
        # (X1,X2 are the intersection points that we want to find)
        a=(d**2+d**2-(2*self.ball_radius)**2)/(2*d)
        # distance from T to C2
        b=d-a
        # distance from C2 to X1 = distance from C2 to X2
        h=np.sqrt(d**2-a**2)

        TC=T-C
        auxiliar_points=(C.T * (b/d)).T+(T.T * (a/d)).T

        intersec1_x=auxiliar_points[:,0]+(h/d)*TC[:,1]
        intersec2_x=auxiliar_points[:,0]-(h/d)*TC[:,1]

        intersec1_y=auxiliar_points[:,1]-(h/d)*TC[:,0]
        intersec2_y=auxiliar_points[:,1]+(h/d)*TC[:,0]

        X1=np.column_stack((intersec1_x,intersec1_y))
        X2=np.column_stack((intersec2_x,intersec2_y))

        return X1, X2
    
    def find_valid_pockets(self,T,P,X1,X2):

        TX1=X1-T
        TX2=X2-T
        rotation=np.cross(TX1, TX2) 
        clockwise=(rotation<0)
        counter_clockwise=(rotation>0)
        
        a = np.cross(P[clockwise] - X1[clockwise], T[clockwise] - X1[clockwise]) > 0
        b = np.cross(P[clockwise] - X2[clockwise], T[clockwise] - X2[clockwise]) < 0
        cw_cond=(a & b)


        X1_counterclock=X2
        X2_counterclock=X1
        a = np.cross(P[counter_clockwise] - X1_counterclock[counter_clockwise], T[counter_clockwise] - X1_counterclock[counter_clockwise]) < 0
        b = np.cross(P[counter_clockwise] - X2_counterclock[counter_clockwise], T[counter_clockwise] - X2_counterclock[counter_clockwise]) > 0
        ccw_cond=(a & b)

        cond=np.full((T.shape[0], ), True)
        cond[clockwise]=cw_cond
        cond[counter_clockwise]=ccw_cond

        return cond

    def find_valid_trajectories(self,origin,destiny,collision_balls):
        
        #For clarity of what is going on we will change nomenclature:
        O=origin
        D=destiny
        E=collision_balls
        OD = D-O
        UOD = OD/np.linalg.norm(OD, axis=1).reshape(-1,1)
        OE = E-O
        DE = E-D
        #collision ball distance to CP line
        distance=abs(np.cross(OE,UOD))
        # sign of dot product of vectors allows to know if collision ball lies "behind" origin
        dot_prod_sign_O=np.sum(OE*UOD,axis=1)
        # sign of dot product of vectors allows to know if collision ball lies "behind" destiny
        dot_prod_sign_D=np.sum(DE*(-UOD),axis=1)
        return (distance < 2*self.ball_radius) & (dot_prod_sign_O > 0) & (dot_prod_sign_D > 0)
            
    def find_geometric_parameters(self,C,T,P):
        """
        Using r, C,T,P we compute the:
            d,b,a distances
            beta, alpha angles
            X point

        Parameters
        ----------    
            C,T,P: tuples
                x, y coordinates of rellevant points
        Returns
        -------
            d,b,a,alpha,beta, X: miscellanous
        """

        r=self.ball_radius
        # we calculate d and b using T and C points
        d=np.linalg.norm(T-C, axis=1)
        b=np.linalg.norm(T-P, axis=1)

        #virtual point X (see fig 4.1 adelaide university thesis)
        # we parametrize the line PT equation and compute the point 
        # that is 2*r distance from T
        t=1+2*r*(1/b)
        x_x=P[:,0]+(T[:,0]-P[:,0])*t
        y_x=P[:,1]+(T[:,1]-P[:,1])*t
        X=np.column_stack([x_x,y_x])

        #To compute a and alpha we need to use cos and sin rules
        beta=np.pi-self.angle_abc(C,T,P)

        a=np.sqrt(d**2+(2*r)**2-2*d*(2*r)*np.cos(beta))
        alpha=np.arcsin(2*r*np.sin(beta)/a)

        return d,b,a,alpha, beta, X

    def setup_pool_frame(self, 
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

        self.mouth_top_middle9 = (x_pocket_top_middle-width_middle, param['vertical_top_offset'])
        self.mouth_top_middle10 = (x_pocket_top_middle+width_middle, param['vertical_top_offset'])

        self.mouth_bottom_middle11 = (x_pocket_top_middle-width_middle, H-param['vertical_bottom_offset'])
        self.mouth_bottom_middle12 = (x_pocket_top_middle+width_middle, H-param['vertical_bottom_offset'])

        cv2.line(img, self.mouth_top_left2, self.mouth_top_right3, (0, 255, 0), thickness=3)
        cv2.line(img, self.mouth_top_right4, self.mouth_bottom_right5, (0, 255, 0), thickness=3)
        cv2.line(img, self.mouth_bottom_right6, self.mouth_bottom_left7, (0, 255, 0), thickness=3)
        cv2.line(img, self.mouth_bottom_left8, self.mouth_top_left1, (0, 255, 0), thickness=3)

        cv2.line(img, self.mouth_top_left1, self.mouth_top_left2, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_top_right3, self.mouth_top_right4, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_bottom_right5, self.mouth_bottom_right6, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_bottom_left7, self.mouth_bottom_left8, (255, 255, 0), thickness=20)

        cv2.line(img, self.mouth_top_middle9, self.mouth_top_middle10, (255, 255, 0), thickness=20)
        cv2.line(img, self.mouth_bottom_middle11, self.mouth_bottom_middle12, (255, 255, 0), thickness=20)

        return img
    

    def setup_pockets(self, precision):

        self.valid_points_mouth_top_left=self.get_equidistant_points(self.mouth_top_left1, self.mouth_top_left2, precision)
        self.valid_points_mouth_top_right=self.get_equidistant_points(self.mouth_top_right3, self.mouth_top_right4, precision)
        self.valid_points_mouth_bottom_right=self.get_equidistant_points(self.mouth_bottom_right5, self.mouth_bottom_right6, precision)
        self.valid_points_mouth_bottom_left=self.get_equidistant_points(self.mouth_bottom_left7, self.mouth_bottom_left8, precision)
        self.valid_points_mouth_top_middle=self.get_equidistant_points(self.mouth_top_middle9, self.mouth_top_middle10, precision)
        self.valid_points_mouth_bottom_middle=self.get_equidistant_points(self.mouth_bottom_middle11, self.mouth_bottom_middle12, precision)
        
        # add ids of pockets:
        pockets_id=np.array([1,3,4,6,2,5])
        pockets_id=np.repeat(pockets_id,precision+1).reshape(-1,1)
        pockets = np.vstack([self.valid_points_mouth_top_left,  #pocket 1      
                            self.valid_points_mouth_top_right,       #pocket 3
                            self.valid_points_mouth_bottom_right,    #pocket 4   
                            self.valid_points_mouth_bottom_left,     #pocket 6  
                            self.valid_points_mouth_top_middle,      #pocket 2 
                            self.valid_points_mouth_bottom_middle,   #pocket 5    
                            ])
        self.pockets=np.hstack([pockets_id,pockets])

    def draw_pool_balls(self, img):
        for ball_num in self.d_centroids:
            x,y=self.d_centroids[ball_num]
            img=cv2.putText(img, "#{}".format(ball_num), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            img=cv2.circle(img, (int(x), int(y)), 8, (255, 0, 255), -1)
            img=cv2.circle(img, (int(x), int(y)), self.ball_radius, (255, 0, 255), 8)

        return img
    


