import cv2
import numpy as np
import pandas as pd
import pool.utils as utils


class Brain: 

    def __init__(self,
                pool_frame,
                ball_radius = None,
        ):
        
        params = utils.Params()
        if ball_radius is None:
            self.ball_radius=params.BALL_RADIUS
        else:
            self.ball_radius=ball_radius
        
        self.pool_frame=pool_frame
    
    def get_all_detected_balls_except_cue(self,d_centroids):

        dict_detected_balls_except_cue=d_centroids.copy()
        dict_detected_balls_except_cue.pop(0, None)
        all_detected_balls_except_cue=np.array([(k, v[0], v[1]) for k, v in dict_detected_balls_except_cue.items()])

        return all_detected_balls_except_cue
    
    def get_balls_to_be_pocket(self,d_centroids,ball_type):
        
        l_detected=[key for key in d_centroids]

        if ball_type=='solid':
            l_type=[1,2,3,4,5,6,7]
        elif ball_type=='strip':
            l_type=[9,10,11,12,13,14,15]

        l_detected_type=list(set(l_type) & set(l_detected))
        dct_detected_type = {key: d_centroids[key] for key in l_detected_type}
        arr_detected_type=np.array([(k, v[0], v[1]) for k, v in dct_detected_type.items()])

        return arr_detected_type

    def get_cue_and_8ball(self,d_centroids):
        if 0 in d_centroids:
            arr_cue = np.array(d_centroids[0])
        else:   
            raise ValueError('cue ball not detected, pool shot cannot be executed')
        if 8 in d_centroids:
            arr_8ball = np.array(d_centroids[8])
        else:   
            raise ValueError('8 ball not detected, game over')

        return arr_cue, arr_8ball
    
    def get_other_balls(self,no_cue,to_pocket):
        result=utils.get_row_combinations_of_two_arrays(to_pocket,no_cue)
        return result
    
    def get_other_balls_twice(self, no_cue, to_pocket):
        result=utils.get_row_combinations_of_two_arrays(to_pocket,to_pocket)
        result=result[result[:,0]!=result[:,3]]
        result=utils.get_row_combinations_of_two_arrays(result,no_cue)

        return result
    
    def get_cue_ball_reflections(self,C): # TODO try to merge get_cue_ball_reflections() and get_T_ball_reflections()
        if len(C.shape)==1:
            C=C.reshape(1,2)

        Cx=C[0,0]
        Cy=C[0,1]

        # check if C inside rect frame
        if Cx>self.pool_frame.right_x:
            Cx=self.pool_frame.right_x
        elif Cx<self.pool_frame.left_x:
            Cx=self.pool_frame.left_x
        if Cy>self.pool_frame.bottom_y:
            Cy=self.pool_frame.bottom_y
        elif Cy<self.pool_frame.top_y:
            Cy=self.pool_frame.top_y

        dist_C_top = Cy-self.pool_frame.top_y
        dist_C_bottom = -Cy+self.pool_frame.bottom_y
        dist_C_left = Cx-self.pool_frame.left_x
        dist_C_right = -Cx+self.pool_frame.right_x

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

        Tx[Tx>self.pool_frame.right_x]=self.pool_frame.right_x
        Tx[Tx<self.pool_frame.left_x]=self.pool_frame.left_x
        Ty[Ty>self.pool_frame.bottom_y]=self.pool_frame.bottom_y
        Ty[Ty<self.pool_frame.top_y]=self.pool_frame.top_y

        dist_T_top = Ty-self.pool_frame.top_y
        dist_T_bottom = -Ty+self.pool_frame.bottom_y
        dist_T_left = Tx-self.pool_frame.left_x
        dist_T_right = -Tx+self.pool_frame.right_x

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
        T_reflect_ids = utils.get_row_combinations_of_two_arrays(T_reflect_sub_id,T_reflect_id)
        T_reflect = np.hstack((T_reflect_ids, T_reflect))
        return T_reflect

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
            
    def find_all_collision_trajectories_in_df(self, df, traj, rows_to_match, collision_balls):
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

    def find_X(self,T,P):

        r=self.ball_radius
        b=np.linalg.norm(T-P, axis=1)
        # virtual point X 
        # we parametrize the line PT equation and compute the point 
        # that is 2*r distance from T
        t=1+2*r*(1/b)
        x_x=P[:,0]+(T[:,0]-P[:,0])*t
        y_x=P[:,1]+(T[:,1]-P[:,1])*t
        X=np.column_stack([x_x,y_x])

        return X
    
    def find_bouncing_points(self,C_reflect, X):
        
        points=np.hstack((C_reflect,X))
        results=np.zeros((points.shape[0], 2))

        #top_segment
        top_point1=self.pool_frame.top_left
        top_point2=self.pool_frame.top_right

        #right_segment
        right_point1=self.pool_frame.top_right
        right_point2=self.pool_frame.bottom_right

        #bottom_segment
        bottom_point1=self.pool_frame.bottom_right
        bottom_point2=self.pool_frame.bottom_left

        #left_segment
        left_point1=self.pool_frame.bottom_left
        left_point2=self.pool_frame.top_left

        #intersection can happen between four different segments (edges of pool frame)

        C_reflect_id=C_reflect[:,0]
        C_reflect_coord=C_reflect[:,1:]
        cond_left_quadrant= (C_reflect_id==4)
        cond_right_quadrant= (C_reflect_id==2)
        cond_top_quadrant= (C_reflect_id==1)
        cond_bottom_quadrant= (C_reflect_id==3)

        results[cond_left_quadrant]=utils.line_intersect(X[cond_left_quadrant],
                                                        C_reflect_coord[cond_left_quadrant],
                                                        left_point1,left_point2)
        results[cond_right_quadrant]=utils.line_intersect(X[cond_right_quadrant],
                                                         C_reflect_coord[cond_right_quadrant],
                                                         right_point1,right_point2)
        results[cond_top_quadrant]=utils.line_intersect(X[cond_top_quadrant],
                                                       C_reflect_coord[cond_top_quadrant],
                                                        top_point1,top_point2)
        results[cond_bottom_quadrant]=utils.line_intersect(X[cond_bottom_quadrant],
                                                        C_reflect_coord[cond_bottom_quadrant],
                                                        bottom_point1,bottom_point2)        
        return results
    
    def deviation_from_ideal_angle(self, T,X,C):
        TX=X-T
        XC=C-X
        angle=utils.angle_between_two_vectors(TX,XC)
        return np.abs(angle)
    
    def find_valid_cushion_impacts(self, B):
        #cushion 1 (horizontal_left)
        xmin1=self.pool_frame.xrange_horizontal_left_cushion[0]
        xmax1=self.pool_frame.xrange_horizontal_left_cushion[1]

        #cushion 2 (horizontal_right)
        xmin2=self.pool_frame.xrange_horizontal_right_cushion[0]
        xmax2=self.pool_frame.xrange_horizontal_right_cushion[1]

        #cushion 3 (vertical)
        ymin=self.pool_frame.xrange_vertical_cushion[0]
        ymax=self.pool_frame.xrange_vertical_cushion[1]
        
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
    
    def actuator_angle(self,df):
        #angle to send to the actuator
        CX = df[['Xx','Xy']].values-df[['Cx','Cy']].values
        x_axis = np.array([[1,0]])
        angle = utils.angle_between_two_vectors(x_axis, CX)
        df['angle'] = np.degrees(angle)
        return df
    
    def ball_config_param(self,d_centroids, ball_type):
        no_cue = self.get_all_detected_balls_except_cue(d_centroids)
        to_pocket = self.get_balls_to_be_pocket(d_centroids,ball_type)
        C, ball8 = self.get_cue_and_8ball(d_centroids)
        C_reflect = self.get_cue_ball_reflections(C)
        T=self.get_other_balls(no_cue,to_pocket)
        TT=self.get_other_balls_twice(no_cue,to_pocket)
        T_reflect=self.get_T_ball_reflections(to_pocket)
        P=self.pool_frame.pockets.pockets
        param={'C':C, 'C_reflect':C_reflect, 'T':T, 'TT':TT, 'T_reflect':T_reflect, 'P':P}
        return param

    def draw_pool_balls(self, d_centroids, img):
        for ball_num in d_centroids:
            x,y=d_centroids[ball_num]
            img=cv2.putText(img, "#{}".format(ball_num), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            img=cv2.circle(img, (int(x), int(y)), 8, (255, 0, 255), -1)
            img=cv2.circle(img, (int(x), int(y)), self.ball_radius, (255, 0, 255), 8)

        return img
    
    def draw_trajectories(self, img, points1, points2):
        
        for point1, point2 in zip(points1, points2): 
            cv2.line(img, point1.astype(int), point2.astype(int), [251, 163, 26], 1) 
        return img


class CTP(Brain):

    def __init__(self,
                 pool_frame,
                 ball_radius = None):
        super().__init__(pool_frame, ball_radius)

    def generate_combinations(self,param):
        comb=utils.get_row_combinations_of_two_arrays(param['T'], param['P'])
        comb=utils.get_row_combinations_of_two_arrays(param['C'], comb)
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
        return df
    
    def collision_trajectories(self,df):
        collision_balls=df[['other_ball_x', 'other_ball_y']].values
        origin1=df[['Cx', 'Cy']].values
        destiny1=df[['Xx', 'Xy']].values
        origin2=df[['Tx', 'Ty']].values
        destiny2=df[['Px', 'Py']].values
        traj=((origin1,destiny1), (origin2, destiny2))
        rows_to_match=['T_id', 'P_id', 'P_sub_id']
        collision_configs=self.find_all_collision_trajectories_in_df(df,traj,rows_to_match, collision_balls)
        return collision_configs
    
    def draw_all_trajectories(self,df,img):
        img=self.draw_trajectories(img,df[['Cx', 'Cy']].values,df[['Xx', 'Xy']].values)
        img=self.draw_trajectories(img,df[['Tx', 'Ty']].values,df[['Px', 'Py']].values) 
            
        return img
    
    def sort_df_by_difficulty(self, df):
        T=df[['Tx', 'Ty']].values
        X=df[['Xx', 'Xy']].values
        C=df[['Cx', 'Cy']].values
        P=df[['Px', 'Py']].values
        XC_TX_abs_angle = self.deviation_from_ideal_angle(T,X,C)
        dist_CX = np.linalg.norm(X-C, axis=1)
        dist_TP = np.linalg.norm(P-T, axis=1)
        df['dificulty'] = dist_CX*dist_TP / np.cos(XC_TX_abs_angle)
        df = df.sort_values(by=['dificulty'], ascending=True)
        return df

    def selected_shots(self,d_centroids,ball_type):
        param=self.ball_config_param(d_centroids,ball_type)
        df=self.generate_combinations(param)
        X_comb = self.find_X(df[['Tx', 'Ty']].values,df[['Px', 'Py']].values)
        df['Xx']=X_comb[:,0]
        df['Xy']=X_comb[:,1]
        collision_configs=self.collision_trajectories(df)
        df=df[~(collision_configs)]
        df = self.sort_df_by_difficulty(df)
        df = self.actuator_angle(df)

        return df

class CBTP(Brain):

    def __init__(self,
                 pool_frame,
                 ball_radius = None):
        super().__init__(pool_frame, ball_radius)

    def generate_combinations(self,param):
        comb=utils.get_row_combinations_of_two_arrays(param['T'], param['P'])
        comb=utils.get_row_combinations_of_two_arrays(param['C_reflect'],comb)
        comb=utils.get_row_combinations_of_two_arrays(param['C'],comb)
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
        return df
    
    def collision_trajectories(self,df):
        collision_balls=df[['other_ball_x', 'other_ball_y']].values
        origin1=df[['Cx', 'Cy']].values
        destiny1=df[['Bx', 'By']].values
        origin2=df[['Bx', 'By']].values
        destiny2=df[['Xx', 'Xy']].values
        origin3=df[['Tx', 'Ty']].values
        destiny3=df[['Px', 'Py']].values
        traj=((origin1,destiny1), (origin2, destiny2), (origin3, destiny3))
        rows_to_match=['T_id', 'C_reflect_id', 'P_id', 'P_sub_id']
        collision_configs=self.find_all_collision_trajectories_in_df(df,traj,rows_to_match, collision_balls)
        return collision_configs
    
    def draw_all_trajectories(self,df,img):
        img=self.draw_trajectories(img,df[['Bx', 'By']].values, df[['Cx', 'Cy']].values)
        img=self.draw_trajectories(img,df[['Bx', 'By']].values, df[['Xx', 'Xy']].values)
        img=self.draw_trajectories(img,df[['Tx', 'Ty']].values, df[['Px', 'Py']].values) 
        return img

    def sort_df_by_difficulty(self, df):
        T=df[['Tx', 'Ty']].values
        X=df[['Xx', 'Xy']].values
        C=df[['Cx', 'Cy']].values
        B=df[['Bx', 'By']].values
        P=df[['Px', 'Py']].values
        XB_TX_abs_angle = self.deviation_from_ideal_angle(T,X,B)
        dist_CB = np.linalg.norm(B-C, axis=1)
        dist_BX = np.linalg.norm(X-B, axis=1)
        dist_TP = np.linalg.norm(P-T, axis=1)
        df['dificulty'] = (dist_CB+dist_BX)*dist_TP / np.cos(XB_TX_abs_angle)
        df = df.sort_values(by=['dificulty'], ascending=True)
        return df

    def selected_shots(self,d_centroids,ball_type):
        param=self.ball_config_param(d_centroids,ball_type)
        df=self.generate_combinations(param)
        X_comb = self.find_X(df[['Tx', 'Ty']].values,
                             df[['Px', 'Py']].values)
        df['Xx']=X_comb[:,0]
        df['Xy']=X_comb[:,1]

        B_comb = self.find_bouncing_points(df[['C_reflect_id','C_reflect_x', 'C_reflect_y']].values,
                                           df[['Xx', 'Xy']].values,)
        df['Bx']=B_comb[:,0]
        df['By']=B_comb[:,1]
        collision_configs=self.collision_trajectories(df)
        df=df[~(collision_configs)]
        valid_bounces=self.find_valid_cushion_impacts(df[['C_reflect_id','Bx','By']].values)
        df=df[valid_bounces]
        df = self.sort_df_by_difficulty(df)
        df = self.actuator_angle(df)
        return df
    
class CTTP(Brain):

    def __init__(self,
                 pool_frame,
                 ball_radius = None):
        super().__init__(pool_frame, ball_radius)

    def generate_combinations(self,param):
        comb=utils.get_row_combinations_of_two_arrays(param['TT'],param['P'])
        comb=utils.get_row_combinations_of_two_arrays(param['C'],comb)
        df=pd.DataFrame({'Cx':comb[:,0],
                        'Cy':comb[:,1],
                        'T_id':comb[:,2],
                        'Tx':comb[:,3],
                        'Ty':comb[:,4],
                        'TT_id':comb[:,5], 
                        'TTx':comb[:,6],   
                        'TTy':comb[:,7],   
                        'other_ball_id':comb[:,8],
                        'other_ball_x':comb[:,9],
                        'other_ball_y':comb[:,10],
                        'P_id':comb[:,11],
                        'P_sub_id':comb[:,12],
                        'Px':comb[:,13],
                        'Py':comb[:,14]})
        return df
    
    def collision_trajectories(self,df):
        collision_balls=df[['other_ball_x', 'other_ball_y']].values
        origin1=df[['Cx', 'Cy']].values
        destiny1=df[['X_new_x', 'X_new_y']].values
        origin2=df[['TTx', 'TTy']].values 
        destiny2=df[['Xx', 'Xy']].values
        origin3=df[['Tx', 'Ty']].values
        destiny3=df[['Px', 'Py']].values
        traj=((origin1,destiny1), (origin2, destiny2), (origin3, destiny3))
        rows_to_match=['T_id', 'TT_id', 'P_id','P_sub_id'] 
        collision_configs=self.find_all_collision_trajectories_in_df(df,traj,rows_to_match, collision_balls)
        return collision_configs
    
    def draw_all_trajectories(self,df,img):
        img=self.draw_trajectories(img,df[['Cx', 'Cy']].values, df[['X_new_x', 'X_new_y']].values)
        img=self.draw_trajectories(img,df[['TTx', 'TTy']].values, df[['Xx', 'Xy']].values)
        img=self.draw_trajectories(img,df[['Tx', 'Ty']].values, df[['Px', 'Py']].values) 
        return img
    
    def sort_df_by_difficulty(self, df):
        T=df[['Tx', 'Ty']].values
        X=df[['Xx', 'Xy']].values
        TT=df[['TTx', 'TTy']].values
        Xnew=df[['X_new_x', 'X_new_y']].values
        C=df[['Cx', 'Cy']].values
        P=df[['Px', 'Py']].values
        XC_TX_abs_angle = self.deviation_from_ideal_angle(T,X,C)
        XnewT_TTXnew_abs_angle = self.deviation_from_ideal_angle(TT,Xnew,T)
        dist_CX = np.linalg.norm(X-C, axis=1)
        dist_TXnew = np.linalg.norm(Xnew-T, axis=1)
        dist_TTP = np.linalg.norm(P-TT, axis=1)
        difficulty1 = dist_CX*dist_TXnew / np.cos(XC_TX_abs_angle)
        difficulty2 = dist_TXnew*dist_TTP / np.cos(XnewT_TTXnew_abs_angle)
        df['dificulty'] = (difficulty1 + difficulty2)/2
        df = df.sort_values(by=['dificulty'], ascending=True)
        return df

    def selected_shots(self,d_centroids,ball_type):
        param=self.ball_config_param(d_centroids,ball_type)
        df=self.generate_combinations(param)
        X_comb = self.find_X(T=df[['Tx', 'Ty']].values,
                            P=df[['Px', 'Py']].values)
        df['Xx']=X_comb[:,0]
        df['Xy']=X_comb[:,1]
        X_new_comb = self.find_X(T=df[['TTx', 'TTy']].values,
                                P=df[['Xx', 'Xy']].values)
        df['X_new_x']=X_new_comb[:,0]
        df['X_new_y']=X_new_comb[:,1]

        collision_configs=self.collision_trajectories(df)
        df=df[~(collision_configs)]
        df = self.sort_df_by_difficulty(df)
        df = self.actuator_angle(df)
        return df
    
class CTBP(Brain):

    def __init__(self,
                 pool_frame,
                 ball_radius = None):
        super().__init__(pool_frame, ball_radius)

    def generate_combinations(self,param):
        comb=utils.get_row_combinations_of_two_arrays(param['T'],param['P'])
        comb=utils.get_row_combinations_of_two_arrays(param['T_reflect'],comb)
        comb=utils.get_row_combinations_of_two_arrays(param['C'],comb)
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
    
    def collision_trajectories(self,df):
        collision_balls=df[['other_ball_x', 'other_ball_y']].values
        origin1=df[['Cx', 'Cy']].values
        destiny1=df[['Xx', 'Xy']].values
        origin2=df[['Tx', 'Ty']].values 
        destiny2=df[['Bx', 'By']].values
        origin3=df[['Bx', 'By']].values
        destiny3=df[['Px', 'Py']].values
        traj=((origin1,destiny1), (origin2, destiny2), (origin3, destiny3))
        rows_to_match=['T_id', 'T_reflect_sub_id', 'P_id', 'P_sub_id']
        collision_configs=self.find_all_collision_trajectories_in_df(df,traj,rows_to_match, collision_balls)
        return collision_configs
    
    def draw_all_trajectories(self,df,img):
        img=self.draw_trajectories(img,df[['Cx', 'Cy']].values, df[['Xx', 'Xy']].values)
        img=self.draw_trajectories(img,df[['Tx', 'Ty']].values, df[['Bx', 'By']].values)
        img=self.draw_trajectories(img,df[['Bx', 'By']].values, df[['Px', 'Py']].values) 
        return img

    def sort_df_by_difficulty(self, df):
        T=df[['Tx', 'Ty']].values
        X=df[['Xx', 'Xy']].values
        C=df[['Cx', 'Cy']].values
        B=df[['Bx', 'By']].values
        P=df[['Px', 'Py']].values
        XC_TX_abs_angle = self.deviation_from_ideal_angle(T,X,C)
        dist_CX = np.linalg.norm(X-C, axis=1)
        dist_TB = np.linalg.norm(B-T, axis=1)
        dist_BP = np.linalg.norm(P-B, axis=1)
        df['dificulty'] = (dist_TB+dist_BP)*dist_CX / np.cos(XC_TX_abs_angle)
        df = df.sort_values(by=['dificulty'], ascending=True)
        return df

    def selected_shots(self,d_centroids,ball_type): 
        param=self.ball_config_param(d_centroids,ball_type)
        df=self.generate_combinations(param)

        B_comb = self.find_bouncing_points(df[['T_reflect_sub_id','T_reflect_x', 'T_reflect_y']].values,
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

        collision_configs=self.collision_trajectories(df)
        df=df[~(collision_configs)]

        valid_bounces=self.find_valid_cushion_impacts(df[['T_reflect_sub_id','Bx','By']].values)
        df=df[valid_bounces]
        df = self.sort_df_by_difficulty(df)
        df = self.actuator_angle(df)
        return df
