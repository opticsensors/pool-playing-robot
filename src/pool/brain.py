import cv2
import numpy as np
import pandas as pd

class Brain(object):
    """
    A class that finds the optimal pool shot 

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
                d_centroids = {},
                ball_radius = None,
        ):
        
        self._d_centroids = d_centroids
        if ball_radius is None:
            self.ball_radius=Brain.BALL_RADIUS
        else:
            self.ball_radius=ball_radius

        self.setup_pool_frame() #TODO -> C/T_reflect use info from this setup !

        self.solid = self.get_balls_to_be_pocket('solid')
        self.strip = self.get_balls_to_be_pocket('strip')
        self.C, self.ball8 = self.get_cue_and_8ball()
        self.C_reflect = self.get_cue_ball_reflections(self.C)
        self.T_solid=self.get_other_balls(self.solid)
        self.T_strip=self.get_other_balls(self.strip)
        self.TT_solid=self.get_other_balls_twice(self.solid,'solid')
        self.TT_strip=self.get_other_balls_twice(self.strip,'strip')
        self.T_reflect_solid=self.get_T_ball_reflections(self.T_solid)
        self.T_reflect_strip=self.get_T_ball_reflections(self.T_strip)


    @property
    def d_centroids(self):
        return self._d_centroids

    @d_centroids.setter #every time ball config changes, we need to update the following attributes
    def d_centroids(self, val):
        self._d_centroids = val
        self.solid = self.get_balls_to_be_pocket('solid')
        self.strip = self.get_balls_to_be_pocket('strip')
        self.C, self.ball8 = self.get_cue_and_8ball()
        self.C_reflect = self.get_cue_ball_reflections(self.C)
        self.T_solid=self.get_other_balls(self.solid)
        self.T_strip=self.get_other_balls(self.strip)
        self.TT_solid=self.get_other_balls_twice(self.solid,'solid')
        self.TT_strip=self.get_other_balls_twice(self.strip,'strip')
        self.T_reflect_solid=self.get_T_ball_reflections(self.T_solid)
        self.T_reflect_strip=self.get_T_ball_reflections(self.T_strip)

    @staticmethod
    def _draw_pocket(img,point,radius):
        cv2.circle(img, point, radius, (0, 255, 255), 8)
        x,y=point
        cv2.line(img, (x+radius, y), (x-radius, y), (0, 255, 255), thickness=8)
        cv2.line(img, (x, y-radius), (x, y+radius), (0, 255, 255), thickness=8)
        return img

    @staticmethod
    def _get_equidistant_points(p1, p2, parts):
        if parts==0:
            points=(p1+p2)/2
        else:
            points_separated=np.linspace(p1[0], p2[0], parts+1),np.linspace(p1[1], p2[1], parts+1)
            points=np.column_stack((points_separated[0],points_separated[1]))
        return points
    
    @staticmethod
    def _angle_between_two_vectors(u,v):
        dot = u[:,0]*v[:,0] + u[:,1]*v[:,1] # equivalent to np.sum(u*v, axis=1)
        cosine_angle = dot / (np.linalg.norm(u, axis=1)* np.linalg.norm(v, axis=1))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    @staticmethod
    def _line_intersect(a1, a2, b1, b2):
        T = np.array([[0, -1], [1, 0]])
        da = np.atleast_2d(a2 - a1)
        db = np.atleast_2d(b2 - b1)
        dp = np.atleast_2d(a1 - b1)
        dap = np.dot(da, T)
        denom = np.sum(dap * db, axis=1)
        num = np.sum(dap * dp, axis=1)
        return np.atleast_2d(num / denom).T * db + b1

    def get_all_detected_balls_except_cue(self):

        dict_detected_balls_except_cue=self.d_centroids.copy()
        dict_detected_balls_except_cue.pop(0, None)
        all_detected_balls_except_cue=np.array([(k, v[0], v[1]) for k, v in dict_detected_balls_except_cue.items()])

        return all_detected_balls_except_cue
    
    def get_balls_to_be_pocket(self,ball_type):
        
        l_detected=[key for key in self.d_centroids]

        if ball_type=='solid':
            l_type=[1,2,3,4,5,6,7]
        elif ball_type=='strip':
            l_type=[9,10,11,12,13,14,15]

        l_detected_type=list(set(l_type) & set(l_detected))
        dct_detected_type = {key: self.d_centroids[key] for key in l_detected_type}
        arr_detected_type=np.array([(k, v[0], v[1]) for k, v in dct_detected_type.items()])

        return arr_detected_type

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

        all_detected_balls_except_cue=self.get_all_detected_balls_except_cue()
        result=self.get_row_combinations_of_two_arrays(T,all_detected_balls_except_cue)
        result=result[result[:,0]!=result[:,3]]
        return result
    
    def get_other_balls_twice(self, T, ball_type):

        all_detected_balls_except_cue=self.get_all_detected_balls_except_cue()
        arr_balls_to_be_pocket=self.get_balls_to_be_pocket(ball_type)

        #detected by specified type
        a=all_detected_balls_except_cue #[:,0].reshape(-1,1) 
        b=arr_balls_to_be_pocket        #[:,0].reshape(-1,1)
        cond=(a[None,:]==b[:,None]).all(-1).any(0)
        all_detected_balls_by_type=all_detected_balls_except_cue[cond]

        result=self.get_row_combinations_of_two_arrays(T,all_detected_balls_by_type)
        result=self.get_row_combinations_of_two_arrays(result,all_detected_balls_except_cue)
        result=result[(result[:,0]!=result[:,3]) & (result[:,0]!=result[:,6]) & (result[:,3]!=result[:,6])]
        return result

    def get_row_combinations_of_two_arrays(self, array1,array2):

        if len(array1.shape)==1:
            array1=array1.reshape(1,2)

        if len(array2.shape)==1:
            array2=array2.reshape(1,2)

        a = np.repeat(array1, array2.shape[0], axis=0)
        b = np.tile(array2, (array1.shape[0],1))
        result = np.hstack([a,b])

        return result

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
    
    def find_if_point_isreachable(self,T,P,X1,X2):

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

    def find_collision_trajectories(self,origin,destiny,collision_balls):
        
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
            
    def find_X(self,T,P):

        r=self.ball_radius
        # we calculate d and b using T and C points
        b=np.linalg.norm(T-P, axis=1)

        #virtual point X (see fig 4.1 adelaide university thesis)
        # we parametrize the line PT equation and compute the point 
        # that is 2*r distance from T
        t=1+2*r*(1/b)
        x_x=P[:,0]+(T[:,0]-P[:,0])*t
        y_x=P[:,1]+(T[:,1]-P[:,1])*t
        X=np.column_stack([x_x,y_x])

        return X
    
    def get_cue_ball_reflections(self,C):
        if len(C.shape)==1:
            C=C.reshape(1,2)

        Cx=C[0,0]
        Cy=C[0,1]

        top_y = self.frame_top_left[1]
        right_x = self.frame_top_right[0]
        bottom_y = self.frame_bottom_right[1]
        left_x = self.frame_bottom_left[0]

        # check if C inside rect frame
        if Cx>right_x:
            Cx=right_x
        elif Cx<left_x:
            Cx=left_x
        if Cy>bottom_y:
            Cy=bottom_y
        elif Cy<top_y:
            Cy=top_y

        dist_C_top = Cy-top_y
        dist_C_bottom = -Cy+bottom_y
        dist_C_left = Cx-left_x
        dist_C_right = -Cx+right_x

        C_top = C + np.array([[0,-2*dist_C_top]])
        C_bottom = C + np.array([[0,2*dist_C_bottom]])
        C_left = C + np.array([[-2*dist_C_left,0]])
        C_right = C + np.array([[2*dist_C_right,0]])
        C_reflect_id=np.array([1,2,3,4]).reshape(-1,1)
        C_reflect_coord= np.vstack((C_top,C_right,C_bottom,C_left))
        C_reflect = np.hstack((C_reflect_id,C_reflect_coord))
        return C_reflect

    def get_T_ball_reflections(self,T):
        Tx=T[:,1].reshape(-1,1)
        Ty=T[:,2].reshape(-1,1)
        T_reflect_sub_id=np.array([1,2,3,4]).reshape(-1,1)
        T_reflect_id = T[:,0].reshape(-1,1)

        top_y = self.frame_top_left[1]
        right_x = self.frame_top_right[0]
        bottom_y = self.frame_bottom_right[1]
        left_x = self.frame_bottom_left[0]

        Tx[Tx>right_x]=right_x
        Tx[Tx<left_x]=left_x
        Ty[Ty>bottom_y]=bottom_y
        Ty[Ty<top_y]=top_y

        dist_T_top = Ty-top_y
        dist_T_bottom = -Ty+bottom_y
        dist_T_left = Tx-left_x
        dist_T_right = -Tx+right_x

        T_top_x = Tx 
        T_top_y = Ty + -2*dist_T_top
        T_top = np.hstack((T_top_x,T_top_y))

        T_bottom_x = Tx 
        T_bottom_y = Ty + 2*dist_T_bottom
        T_bottom = np.hstack((T_bottom_x,T_bottom_y))

        T_left_x = Tx - 2*dist_T_left
        T_left_y = Ty 
        T_left = np.hstack((T_left_x,T_left_y))

        T_right_x = Tx + 2*dist_T_right
        T_right_y = Ty 
        T_right = np.hstack((T_right_x,T_right_y))

        T_reflect = np.vstack((T_top,T_right,T_bottom,T_left))
        T_reflect_ids = self.get_row_combinations_of_two_arrays(T_reflect_sub_id,T_reflect_id)
        T_reflect = np.hstack((T_reflect_ids, T_reflect))
        return T_reflect
    
    def find_bouncing_points(self,C_reflect, X):
        
        points=np.hstack((C_reflect,X))
        results=np.zeros((points.shape[0], 2))

        #top_segment
        top_point1=self.frame_top_left
        top_point2=self.frame_top_right

        #right_segment
        right_point1=self.frame_top_right
        right_point2=self.frame_bottom_right

        #bottom_segment
        bottom_point1=self.frame_bottom_right
        bottom_point2=self.frame_bottom_left

        #left_segment
        left_point1=self.frame_bottom_left
        left_point2=self.frame_top_left

        #intersection can happen between four different segments (edges of pool frame)
        xmin=top_point1[0]
        xmax=top_point2[0]
        ymin=top_point1[1]
        ymax=left_point1[1]

        x=C_reflect[:,0]
        y=C_reflect[:,1]
        cond_left_quadrant= (x<xmin) & (y>ymin) & (y<ymax)
        cond_right_quadrant= (x>xmax) & (y>ymin) & (y<ymax)
        cond_top_quadrant= (x>xmin) & (x<xmax) & (y<ymin)
        cond_bottom_quadrant= (x>xmin) & (x<xmax) & (y>ymax)

        results[cond_left_quadrant]=self._line_intersect(X[cond_left_quadrant],
                                                        C_reflect[cond_left_quadrant],
                                                        left_point1,left_point2)
        results[cond_right_quadrant]=self._line_intersect(X[cond_right_quadrant],
                                                         C_reflect[cond_right_quadrant],
                                                         right_point1,right_point2)
        results[cond_top_quadrant]=self._line_intersect(X[cond_top_quadrant],
                                                       C_reflect[cond_top_quadrant],
                                                        top_point1,top_point2)
        results[cond_bottom_quadrant]=self._line_intersect(X[cond_bottom_quadrant],
                                                          C_reflect[cond_bottom_quadrant],
                                                        bottom_point1,bottom_point2)        
        return results

    def deviation_from_ideal_angle(self, T,X,C):
        TX=X-T
        XC=C-X
        angle=self._angle_between_two_vectors(TX,XC)
        return np.abs(angle)
    
    def find_invalid_cushion_impacts(self, B):
        #cushion 1 (horizontal_left)
        xmin1=self.xmin_horizontal_left_cushion
        xmax1=self.xmax_horizontal_left_cushion

        #cushion 2 (horizontal_right)
        xmin2=self.xmin_horizontal_right_cushion
        xmax2=self.xmax_horizontal_right_cushion

        #cushion 3 (vertical)
        ymin=self.ymin_vertical_cushion
        ymax=self.ymax_vertical_cushion
        
        cond_horiz = (B[:,0]==1) | (B[:,0]==3)
        cond_verti = (B[:,0]==2) | (B[:,0]==4)
        cond_between_x1 = (B[:,1]>xmin1) & (B[:,1]<xmax1)
        cond_between_x2 = (B[:,1]>xmin2) & (B[:,1]<xmax2)
        cond_between_y = (B[:,2]>ymin) & (B[:,2]<ymax)
        
        cond1=cond_horiz & cond_between_x1
        cond2=cond_horiz & cond_between_x2
        cond3=cond_verti & cond_between_y

        final_cond = cond1 | cond2 | cond3

        return final_cond

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

        self.frame_top_left = np.array((param['horizontal_left_offset'], param['vertical_top_offset']))
        self.mouth_top_left1 = np.array((param['horizontal_left_offset'], param['vertical_top_offset']+width_corner))
        self.mouth_top_left2 = np.array((param['horizontal_left_offset'] + width_corner, param['vertical_top_offset']))

        self.frame_top_right = np.array((W-param['horizontal_right_offset'], param['vertical_top_offset']))
        self.mouth_top_right3 = np.array((W-param['horizontal_right_offset']-width_corner, param['vertical_top_offset']))
        self.mouth_top_right4 = np.array((W-param['horizontal_right_offset'], param['vertical_top_offset']+width_corner))

        self.frame_bottom_right = np.array((W-param['horizontal_right_offset'], H-param['vertical_bottom_offset']))
        self.mouth_bottom_right5 = np.array((W-param['horizontal_right_offset'], H-param['vertical_bottom_offset']-width_corner))
        self.mouth_bottom_right6 = np.array((W-param['horizontal_right_offset']-width_corner, H-param['vertical_bottom_offset']))

        self.frame_bottom_left = np.array((param['horizontal_left_offset'], H-param['vertical_bottom_offset']))
        self.mouth_bottom_left7 = np.array((param['horizontal_left_offset']+width_corner, H-param['vertical_bottom_offset']))
        self.mouth_bottom_left8 = np.array((param['horizontal_left_offset'], H-param['vertical_bottom_offset']-width_corner))

        self.mouth_top_middle9 = np.array((x_pocket_top_middle-width_middle, param['vertical_top_offset']))
        self.mouth_top_middle10 = np.array((x_pocket_top_middle+width_middle, param['vertical_top_offset']))

        self.mouth_bottom_middle11 = np.array((x_pocket_top_middle-width_middle, H-param['vertical_bottom_offset']))
        self.mouth_bottom_middle12 = np.array((x_pocket_top_middle+width_middle, H-param['vertical_bottom_offset']))

        self.xmin_horizontal_left_cushion=445
        self.xmax_horizontal_left_cushion=2154
        self.xmin_horizontal_right_cushion=2729
        self.xmax_horizontal_right_cushion=4338
        self.ymin_vertical_cushion=476
        self.ymax_vertical_cushion=2223

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

        self.valid_points_mouth_top_left=self._get_equidistant_points(self.mouth_top_left1, self.mouth_top_left2, precision)
        self.valid_points_mouth_top_right=self._get_equidistant_points(self.mouth_top_right3, self.mouth_top_right4, precision)
        self.valid_points_mouth_bottom_right=self._get_equidistant_points(self.mouth_bottom_right5, self.mouth_bottom_right6, precision)
        self.valid_points_mouth_bottom_left=self._get_equidistant_points(self.mouth_bottom_left7, self.mouth_bottom_left8, precision)
        self.valid_points_mouth_top_middle=self._get_equidistant_points(self.mouth_top_middle9, self.mouth_top_middle10, precision)
        self.valid_points_mouth_bottom_middle=self._get_equidistant_points(self.mouth_bottom_middle11, self.mouth_bottom_middle12, precision)
        
        # add ids of pockets:
        pockets_id=np.array([1,3,4,6,2,5]).reshape(-1,1)
        pockets_sub_id=np.arange(1,precision+2).reshape(-1,1)
        pocket_ids=self.get_row_combinations_of_two_arrays(pockets_id,pockets_sub_id)
        pockets = np.vstack([self.valid_points_mouth_top_left,  #pocket 1      
                            self.valid_points_mouth_top_right,       #pocket 3
                            self.valid_points_mouth_bottom_right,    #pocket 4   
                            self.valid_points_mouth_bottom_left,     #pocket 6  
                            self.valid_points_mouth_top_middle,      #pocket 2 
                            self.valid_points_mouth_bottom_middle,   #pocket 5    
                            ])
        self.pockets=np.hstack([pocket_ids,pockets])
        self.P=self.pockets

    def draw_pool_balls(self, img):
        for ball_num in self.d_centroids:
            x,y=self.d_centroids[ball_num]
            img=cv2.putText(img, "#{}".format(ball_num), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            img=cv2.circle(img, (int(x), int(y)), 8, (255, 0, 255), -1)
            img=cv2.circle(img, (int(x), int(y)), self.ball_radius, (255, 0, 255), 8)

        return img
    
    def draw_trajectories(self, img, points1,points2):
        
        for point1, point2 in zip(points1, points2): 
            cv2.line(img, point1.astype(int), point2.astype(int), [251, 163, 26], 1) 
        return img

    #################################################################################################################################################################
    
    def choose_param_based_on_ball_type(self,ball_type):
        if ball_type=='solid':
            return self.C,self.T_solid,self.TT_solid,self.P,self.C_reflect,self.T_reflect_solid
        elif ball_type=='strip':
            return self.C,self.T_strip,self.TT_strip,self.P,self.C_reflect,self.T_reflect_strip
    
    def wrapper_generate_combinations(self,shot_type,C,T,TT,P,C_reflect,T_reflect):
        
        if shot_type=='CTP':
            comb=self.get_row_combinations_of_two_arrays(T,P)
            comb=self.get_row_combinations_of_two_arrays(C,comb)
            df=pd.DataFrame({'Cx':comb[:,0],
                            'Cy':comb[:,1],
                            'T_id':comb[:,2],
                            'Tx':comb[:,3],
                            'Ty':comb[:,4],
                            'other_ball_id':comb[:,5],
                            'other_ball_x':comb[:,6],
                            'other_ball_y':comb[:,7],
                            'P_id':comb[:,8],
                            'P_sub_id':comb[:,9],
                            'Px':comb[:,10],
                            'Py':comb[:,11]})
            
        elif shot_type=='CBTP':
            comb=self.get_row_combinations_of_two_arrays(T,P)
            comb=self.get_row_combinations_of_two_arrays(C_reflect,comb)
            comb=self.get_row_combinations_of_two_arrays(C,comb)
            df=pd.DataFrame({'Cx':comb[:,0],
                            'Cy':comb[:,1],
                            'C_reflect_id': comb[:,2],
                            'C_reflect_x':comb[:,3],
                            'C_reflect_y':comb[:,4],
                            'T_id':comb[:,5],
                            'Tx':comb[:,6],
                            'Ty':comb[:,7],
                            'other_ball_id':comb[:,8],
                            'other_ball_x':comb[:,9],
                            'other_ball_y':comb[:,10],
                            'P_id':comb[:,11],
                            'P_sub_id':comb[:,12],
                            'Px':comb[:,13],
                            'Py':comb[:,14]})

        elif shot_type=='CTTP':
            comb=self.get_row_combinations_of_two_arrays(TT,P)
            comb=self.get_row_combinations_of_two_arrays(C,comb)
            df=pd.DataFrame({'Cx':comb[:,0],
                            'Cy':comb[:,1],
                            'T_id':comb[:,2],
                            'Tx':comb[:,3],
                            'Ty':comb[:,4],
                            'B_id':comb[:,5],
                            'Bx':comb[:,6],
                            'By':comb[:,7],
                            'other_ball_id':comb[:,8],
                            'other_ball_x':comb[:,9],
                            'other_ball_y':comb[:,10],
                            'P_id':comb[:,11],
                            'P_sub_id':comb[:,12],
                            'Px':comb[:,13],
                            'Py':comb[:,14]})

        elif shot_type=='CTBP':
            comb=self.get_row_combinations_of_two_arrays(T,P)
            comb=self.get_row_combinations_of_two_arrays(T_reflect,comb)
            comb=self.get_row_combinations_of_two_arrays(C,comb)
            df=pd.DataFrame({'Cx':comb[:,0],
                            'Cy':comb[:,1],
                            'T_reflect_id': comb[:,3],
                            'T_reflect_sub_id': comb[:,2],
                            'T_reflect_x':comb[:,4],
                            'T_reflect_y':comb[:,5],
                            'T_id':comb[:,6],
                            'Tx':comb[:,7],
                            'Ty':comb[:,8],
                            'other_ball_id':comb[:,9],
                            'other_ball_x':comb[:,10],
                            'other_ball_y':comb[:,11],
                            'P_id':comb[:,12],
                            'P_sub_id':comb[:,13],
                            'Px':comb[:,14],
                            'Py':comb[:,15]})
            df=df[df['T_reflect_id']==df['T_id']]
        return df

    def wrapper_valid_trajectories(self,origin,end,point_to_check):
        X1_comb,X2_comb = self.find_X1_and_X2(origin,end)
        valid_traj = self.find_if_point_isreachable(end,point_to_check,X1_comb,X2_comb)
        return valid_traj
    
    def wrapper_collision_trajectories(self,shot_type,df):
        
        collision_balls=df[['other_ball_x', 'other_ball_y']].values

        if shot_type=='CTP':
            origin1=df[['Cx', 'Cy']].values
            destiny1=df[['Xx', 'Xy']].values
            origin2=df[['Tx', 'Ty']].values
            destiny2=df[['Px', 'Py']].values
            traj=((origin1,destiny1), (origin2, destiny2))
            rows_to_match=['T_id', 'P_id', 'P_sub_id']
        
        elif shot_type=='CBTP':
            origin1=df[['Cx', 'Cy']].values
            destiny1=df[['Bx', 'By']].values
            origin2=df[['Bx', 'By']].values
            destiny2=df[['Xx', 'Xy']].values
            origin3=df[['Tx', 'Ty']].values
            destiny3=df[['Px', 'Py']].values
            traj=((origin1,destiny1), (origin2, destiny2), (origin3, destiny3))
            rows_to_match=['T_id', 'P_id', 'P_sub_id']
        
        elif shot_type=='CTTP':
            origin1=df[['Cx', 'Cy']].values
            destiny1=df[['X_new_x', 'X_new_y']].values
            origin2=df[['Bx', 'By']].values #TODO -> convert to TTx, TTy
            destiny2=df[['Xx', 'Xy']].values
            origin3=df[['Tx', 'Ty']].values
            destiny3=df[['Px', 'Py']].values
            traj=((origin1,destiny1), (origin2, destiny2), (origin3, destiny3))
            rows_to_match=['T_id', 'B_id', 'P_id','P_sub_id'] #TODO -> convert to B_id, TT_id

        elif shot_type=='CTBP':
            origin1=df[['Cx', 'Cy']].values
            destiny1=df[['Xx', 'Xy']].values
            origin2=df[['Tx', 'Ty']].values 
            destiny2=df[['Bx', 'By']].values
            origin3=df[['Bx', 'By']].values
            destiny3=df[['Px', 'Py']].values
            traj=((origin1,destiny1), (origin2, destiny2), (origin3, destiny3))
            rows_to_match=['T_id', 'P_id', 'P_sub_id']

        list_collision_configs=[]
        for origin,destiny in traj:
            collision_origin_destiny=self.find_collision_trajectories(origin,destiny,collision_balls)
            list_collision_configs.append(collision_origin_destiny)        

        collision_configs=np.bitwise_or.reduce(list_collision_configs)
        df_collisions=df[collision_configs]
        arr_collisions=df_collisions[rows_to_match].values
        arr_configs=df[rows_to_match].values
        collision_configs=(arr_configs[None,:]==arr_collisions[:,None]).all(-1).any(0)
        
        return collision_configs

    def CTP_shots(self,ball_type):
        C,T,TT,P,C_reflect,T_reflect=self.choose_param_based_on_ball_type(ball_type)
        df=self.wrapper_generate_combinations('CTP',C,T,TT,P,C_reflect,T_reflect)
        valid_pockets=self.wrapper_valid_trajectories(origin=df[['Cx', 'Cy']].values,
                                                      end=df[['Tx', 'Ty']].values,
                                                      point_to_check=df[['Px', 'Py']].values)
        df=df[valid_pockets]
        X_comb = self.find_X(df[['Tx', 'Ty']].values,df[['Px', 'Py']].values)
        df['Xx']=X_comb[:,0]
        df['Xy']=X_comb[:,1]
        collision_configs=self.wrapper_collision_trajectories(df,'CTP')
        df=df[~(collision_configs)]
        df['XC_TX_abs_angle'] = self.deviation_from_ideal_angle(df[['Tx', 'Ty']].values,
                                                                df[['Xx', 'Xy']].values,
                                                                df[['Cx', 'Cy']].values)
        df=df[df['XC_TX_abs_angle'] < 60]
        return df


    def CBTP_shots(self,ball_type):

        C,T,TT,P,C_reflect,T_reflect=self.choose_param_based_on_ball_type(ball_type)
        df=self.wrapper_generate_combinations('CBTP',C,T,TT,P,C_reflect,T_reflect)
        valid_pockets=self.wrapper_valid_trajectories(origin=df[['Cx', 'Cy']].values,
                                                      end=df[['Tx', 'Ty']].values,
                                                      point_to_check=df[['Px', 'Py']].values)
        df=df[valid_pockets]
        X_comb = self.find_X(df[['Tx', 'Ty']].values,
                             df[['Px', 'Py']].values)
        df['Xx']=X_comb[:,0]
        df['Xy']=X_comb[:,1]
        B_comb = self.find_bouncing_points(df[['C_reflect_x', 'C_reflect_y']].values,
                                           df[['Xx', 'Xy']].values,)
        df['Bx']=B_comb[:,0]
        df['By']=B_comb[:,1]
        collision_configs=self.wrapper_collision_trajectories(df,'CBTP')
        df=df[~(collision_configs)]
        df['XB_TX_abs_angle'] = self.deviation_from_ideal_angle(df[['Tx', 'Ty']].values,
                                                                df[['Xx', 'Xy']].values,
                                                                df[['Bx', 'By']].values)
        df=df[df['XB_TX_abs_angle'] < 50]
        valid_bounces=self.find_invalid_cushion_impacts(df[['C_reflect_id','Bx','By']].values)
        df=df[valid_bounces]

        return df
    
    def CTTP_shots(self,ball_type):
        C,T,TT,P,C_reflect,T_reflect=self.choose_param_based_on_ball_type(ball_type)
        df=self.wrapper_generate_combinations('CTTP',C,T,TT,P,C_reflect,T_reflect)
        valid_pockets=self.wrapper_valid_trajectories(origin=df[['Bx', 'By']].values,
                                                      end=df[['Tx', 'Ty']].values,
                                                      point_to_check=df[['Px', 'Py']].values)
        df=df[valid_pockets]
        X_comb = self.find_X(T=df[['Tx', 'Ty']].values,
                            P=df[['Px', 'Py']].values)
        df['Xx']=X_comb[:,0]
        df['Xy']=X_comb[:,1]
        X_new_comb = self.find_X(T=df[['Bx', 'By']].values,
                                P=df[['Xx', 'Xy']].values)
        df['X_new_x']=X_new_comb[:,0]
        df['X_new_y']=X_new_comb[:,1]
        valid_pockets=self.wrapper_valid_trajectories(origin=df[['Cx', 'Cy']].values,
                                                      end=df[['Bx', 'By']].values,
                                                      point_to_check=df[['Xx', 'Xy']].values)
        df=df[valid_pockets]
        collision_configs=self.wrapper_collision_trajectories(df,'CTTP')
        df=df[~(collision_configs)]
        df['XB_TX_abs_angle'] = self.deviation_from_ideal_angle(df[['Tx', 'Ty']].values,
                                                                df[['Xx', 'Xy']].values,
                                                                df[['Bx', 'By']].values)
        df=df[df['XB_TX_abs_angle'] < 45]
        df['XC_BX_abs_angle'] = self.deviation_from_ideal_angle(df[['Bx', 'By']].values,
                                                                df[['X_new_x', 'X_new_y']].values,
                                                                df[['Cx', 'Cy']].values)
        df=df[df['XC_BX_abs_angle'] < 45]

        return df

    def CTBP_shots(self,ball_type):
        C,T,TT,P,C_reflect,T_reflect=self.choose_param_based_on_ball_type(ball_type)
        df=self.wrapper_generate_combinations('CTBP',C,T,TT,P,C_reflect,T_reflect)
        B_comb = self.find_bouncing_points(df[['T_reflect_x', 'T_reflect_y']].values,
                                            df[['Px', 'Py']].values,)
        df['Bx']=B_comb[:,0]
        df['By']=B_comb[:,1]
        df=df[((df['P_id']==6) & (df['T_reflect_sub_id']==1)) |
        ((df['P_id']==6) & (df['T_reflect_sub_id']==2)) |
        ((df['P_id']==5) & (df['T_reflect_sub_id']==1)) |
        ((df['P_id']==5) & (df['T_reflect_sub_id']==2)) |
        ((df['P_id']==5) & (df['T_reflect_sub_id']==4)) |
        ((df['P_id']==4) & (df['T_reflect_sub_id']==1)) |
        ((df['P_id']==4) & (df['T_reflect_sub_id']==4)) |
        ((df['P_id']==3) & (df['T_reflect_sub_id']==3)) |
        ((df['P_id']==3) & (df['T_reflect_sub_id']==4)) |
        ((df['P_id']==2) & (df['T_reflect_sub_id']==2)) |
        ((df['P_id']==2) & (df['T_reflect_sub_id']==3)) |
        ((df['P_id']==2) & (df['T_reflect_sub_id']==4)) |
        ((df['P_id']==1) & (df['T_reflect_sub_id']==3)) |
        ((df['P_id']==1) & (df['T_reflect_sub_id']==2)) ]

        X_comb = self.find_X(  df[['Tx', 'Ty']].values,
                                df[['Bx', 'By']].values)
        df['Xx']=X_comb[:,0]
        df['Xy']=X_comb[:,1]

        valid_pockets=self.wrapper_valid_trajectories(origin=df[['Cx', 'Cy']].values,
                                                      end=df[['Tx', 'Ty']].values,
                                                      point_to_check=df[['Bx', 'By']].values)
        df=df[valid_pockets]
        collision_configs=self.wrapper_collision_trajectories(df,'CTBP')
        df=df[~(collision_configs)]

        valid_bounces=self.find_invalid_cushion_impacts(df[['C_reflect_id','Bx','By']].values)
        df=df[valid_bounces]

        return df