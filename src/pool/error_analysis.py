import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.image as mpimg
from pool import utils

class ErrorAnalysis:
    
    def __init__(self, 
                 pockets=None,
                 ball_radius=None,
                 pool_table_size=None
                 ):
        params=utils.Params()
        if pockets is None:
            self.pockets=np.array(params.POCKETS_CM) 
        else: 
            self.pockets=pockets
        if ball_radius is None:
            self.ball_radius=params.BALL_RADIUS_CM
        else:
            self.ball_radius=ball_radius
        if pool_table_size is None:
            self.pool_table_size=params.DISPLAY_SIZE_CM
        else:
            self.pool_table_size=pool_table_size

    def generate_combinations(self, num_real_points, num_estimated_points, maximum_error_vision_system):
        safety_distance=3*self.ball_radius #so the balls dont touch the pool table walls
        #real points configs
        C=utils.generate_random_numbers_inside_rectangle(num_real_points,self.pool_table_size,safety_distance)
        T=utils.generate_random_numbers_inside_rectangle(num_real_points,self.pool_table_size,safety_distance)
        index_real = np.arange(num_real_points)
        CT = np.c_[index_real, C, T]
        #estimated points configs
        index_estimated = np.arange(num_estimated_points).reshape(-1,1)
        CT_extended=utils.get_row_combinations_of_two_arrays(CT, index_estimated)
        C_extended = CT_extended[:,1:3]
        T_extended = CT_extended[:,3:5]
        C_estimated=utils.generate_random_number_inside_circle(C_extended,maximum_error_vision_system)
        T_estimated=utils.generate_random_number_inside_circle(T_extended,maximum_error_vision_system)
        CT_real_and_estimated = np.c_[CT_extended, C_estimated, T_estimated]
        #pockets configs
        index_pocket = np.arange(1,7)
        pockets = np.c_[index_pocket,self.pockets]
        comb = utils.get_row_combinations_of_two_arrays(CT_real_and_estimated, pockets)
        df=pd.DataFrame({'real_point_id':comb[:,0],
                        'C_x':comb[:,1],
                        'C_y':comb[:,2],
                        'T_x':comb[:,3],
                        'T_y':comb[:,4],
                        'estimated_point_id':comb[:,5],
                        'C_estimated_x':comb[:,6],
                        'C_estimated_y':comb[:,7],
                        'T_estimated_x':comb[:,8],
                        'T_estimated_y':comb[:,9],
                        'pocket_id':comb[:,10],
                        'P_x':comb[:,11],
                        'P_y':comb[:,12]})
        return df

    def compute_geometric_parameters(self, df):
        """

        """
        C=df[['C_x','C_y']].values
        T=df[['T_x','T_y']].values
        P=df[['P_x','P_y']].values
        r=self.ball_radius

        # we calculate d and b using T and C points
        d=np.linalg.norm(T-C, axis=1)
        b=np.linalg.norm(T-P, axis=1)

        #To compute a and alpha we need to use cos and sin rules
        beta=np.pi-utils.angle_between_3_points(C,T,P)
        #beta=angle_abc(C_arr,T_arr,X_arr)
        #phi=angle_abc(T_arr,C_arr,np.array([1,0]))
        a=np.sqrt(d**2+(2*r)**2-2*d*(2*r)*np.cos(beta))
        alpha=np.arcsin(2*r*np.sin(beta)/a)
        #alpha=angle_abc(X_arr,C_arr,T_arr)

        df['b']=b
        df['a']=a
        df['d']=d
        df['beta']=beta
        df['alpha']=alpha

        return df
    
    def compute_X(self, df):

        T=df[['T_x','T_y']].values
        P=df[['P_x','P_y']].values
        b=np.linalg.norm(T-P, axis=1)
        r=self.ball_radius

        # virtual point X 
        # we parametrize the line PT equation and compute the point 
        # that is 2*r distance from T
        t=1+2*r*(1/b)
        x_x=P[:,0]+(T[:,0]-P[:,0])*t
        y_x=P[:,1]+(T[:,1]-P[:,1])*t
        X=np.column_stack([x_x,y_x])        
        df['X_x']=X[:,0]
        df['X_y']=X[:,1]
        return df
    
    def compute_X_estimated(self, df):

        T_estimated=df[['T_estimated_x','T_estimated_y']].values
        P=df[['P_x','P_y']].values
        b=np.linalg.norm(T_estimated-P, axis=1)
        r=self.ball_radius

        # virtual point X 
        # we parametrize the line PT equation and compute the point 
        # that is 2*r distance from T
        t=1+2*r*(1/b)
        x_x=P[:,0]+(T_estimated[:,0]-P[:,0])*t
        y_x=P[:,1]+(T_estimated[:,1]-P[:,1])*t
        X_estimated=np.column_stack([x_x,y_x])        
        df['X_estimated_x']=X_estimated[:,0]
        df['X_estimated_y']=X_estimated[:,1]
        return df

    def valid_CT_points(self, df):
        df[df['d']>2.5*self.ball_radius]
        return df

    def cue_ball_trajectory(self, df):
        """
        
        """
        C=df[['C_x','C_y']].values
        C_estimated=df[['C_estimated_x','C_estimated_y']].values
        X_estimated=df[['X_estimated_x','X_estimated_y']].values
        r=self.ball_radius
        
        #line parallel to C'X' that goes trough C
        slope=(X_estimated[:,1]-C_estimated[:,1])/(X_estimated[:,0]-C_estimated[:,0])
        intercept=C[:,1]-slope*C[:,0]

        return slope,intercept


    def compute_Q(self, df): #TODO vectorize

        C=df[['C_x','C_y']].values
        T=df[['T_x','T_y']].values
        P=df[['P_x','P_y']].values
        r=self.ball_radius

        slope, intercept = self.cue_ball_trajectory(df)

        X_calculated1,X_calculated2=utils.intersection_circle_line(slope, intercept,r,T)

        distCX_calculated1 = np.linalg.norm(C-X_calculated1, axis=1)
        distCX_calculated2 = np.linalg.norm(C-X_calculated2, axis=1)

        # we choose the point of intersection that is closest to C
        Cond1 = (distCX_calculated1<distCX_calculated2)
        Cond2 = ~Cond1 # distCX_calculated2<distCX_calculated1
        X_calculated = np.zeros_like(C)
        X_calculated[Cond1] = X_calculated1[Cond1]
        X_calculated[Cond2] = X_calculated2[Cond2]
        df['X_calculated_x'] = X_calculated[:,0]
        df['X_calculated_y'] = X_calculated[:,1]

        #line X''T (1)
        slope1=(X_calculated[:,1]-T[:,1])/(X_calculated[:,0]-T[:,0])
        intercept1=X_calculated[:,1]-slope1*X_calculated[:,0]

        #line TP (=line XT) (2)
        slope2=(P[:,1]-T[:,1])/(P[:,0]-T[:,0])
        intercept2=T[:,1]-slope2*T[:,0]

        #line perpendicular to TP (3)
        slope3=-1/slope2
        intercept3=P[:,1]-slope3*P[:,0]

        #intersection between line X''T and line perpendicular to TP
        Qx=(intercept1-intercept3)/(slope3-slope1)
        Qy=slope1*Qx+intercept1
        df['Q_x']=Qx
        df['Q_y']=Qy
        Q=np.c_[Qx,Qy]
        delta=np.linalg.norm(Q-P, axis=1)
        df['delta']=delta

        return df

    def pockets_inside_region_of_interest(self,df):

        C=df[['C_x','C_y']].values
        T=df[['T_x','T_y']].values
        P=df[['P_x','P_y']].values
        two_times_r=2*self.ball_radius
        d=df['d'].values

        # two_times_r=2*r
        M=(C+T)/2    
        X1,X2=utils.intersection_two_circles(M,T,d/2,two_times_r)

        #now we need to compute the lines X1T and X2T
        #line X1T (1)
        slope1=(X1[:,1]-T[:,1])/(X1[:,0]-T[:,0])
        intercept1=X1[:,1]-slope1*X1[:,0]
        df['slope_X1T']=slope1
        df['intercept_X1T']=intercept1

        #line X2T (2)
        slope2=(X2[:,1]-T[:,1])/(X2[:,0]-T[:,0])
        intercept2=X2[:,1]-slope2*X2[:,0]
        df['slope_X2T']=slope2
        df['intercept_X2T']=intercept2

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

        return df[cond]
    
    def get_error_data(self,
                       num_real_points,
                       num_estimated_points,
                       maximum_error_vision_system):

        df=self.generate_combinations(num_real_points, 
                                    num_estimated_points, 
                                    maximum_error_vision_system)
        df=self.compute_geometric_parameters(df)
        df=self.compute_X(df)
        df=self.compute_X_estimated(df)
        df=self.pockets_inside_region_of_interest(df)
        df=self.valid_CT_points(df)
        df=self.compute_Q(df)

        return df

class DebugErrorAnalysis(ErrorAnalysis):

    def __init__(self,
                 pockets=None,
                 ball_radius=None,
                 pool_table_size=None):
        super().__init__(pockets, ball_radius,pool_table_size)

        params=utils.Params()
        path_to_repo=params.PATH_REPO
        self.img = mpimg.imread(os.path.join(path_to_repo,'data','pool_table.png'))

    def draw_ideal_configuration(self,ax,row_df):
        C=(row_df['C_x'], row_df['C_y'])
        T=(row_df['T_x'], row_df['T_y'])
        P=(row_df['P_x'], row_df['P_y'])
        X=(row_df['X_x'], row_df['X_y'])
        r=self.ball_radius

        #plot geometric situation
        ax.add_patch(plt.Circle(C, r, color='b',fill=False, clip_on=False,linewidth=0.5))
        ax.add_patch(plt.Circle(T, r, color='r',fill=False, clip_on=False,linewidth=0.5))
        ax.add_patch(plt.Circle(X, r, color='b',fill=False, clip_on=False,linewidth=0.5))
        ax.add_artist(plt.Circle(P, 0.5, color='b',fill=True, clip_on=False,linewidth=0.5))
        ax.add_artist(plt.Circle(C, 0.1, color='b',fill=True, clip_on=False,linewidth=0.5))
        ax.add_artist(plt.Circle(T, 0.1, color='r',fill=True, clip_on=False,linewidth=0.5))
        ax.add_patch(plt.Circle(T, 2*r, color='r',fill=False, clip_on=False,linewidth=0.5,linestyle='--'))
        ax.add_artist(plt.Circle(X, 0.1, color='b',fill=True, clip_on=False,linewidth=0.5))

        lines = [[C, X], [X, T], [T, P]]
        lc = mc.LineCollection(lines, colors='b', linewidths=0.5)
        ax.add_collection(lc)

        #Use adjustable='box-forced' to make the plot area square-shaped as well.
        ax.set_aspect('equal', adjustable='box')
        
        return ax

    def draw_real_configuration(self,ax,row_df):
        C=(row_df['C_x'], row_df['C_y'])
        X_calculated=(row_df['X_calculated_x'], row_df['X_calculated_y'])
        Q=(row_df['Q_x'], row_df['Q_y'])
        r=self.ball_radius

        #plot geometric situation
        ax.add_patch(plt.Circle(X_calculated, r, color='orange',fill=False, clip_on=False,linewidth=0.5,linestyle='--'))
        ax.add_artist(plt.Circle(X_calculated, 0.1, color='orange',fill=True, clip_on=False,linewidth=0.5,linestyle='--'))
        ax.add_artist(plt.Circle(Q, 0.5, color='orange',fill=True, clip_on=False,linewidth=0.5))

        lines = [[C, X_calculated], [X_calculated, Q]]
        lc = mc.LineCollection(lines, colors='orange', linewidths=0.5)
        ax.add_collection(lc)

        #Use adjustable='box-forced' to make the plot area square-shaped as well.
        ax.set_aspect('equal', adjustable='box')
        
        return ax

    def draw_pool_table_with_pockets(self,ax):
        #ax.add_patch(plt.Rectangle((0, 0), W, H,color='k',fill=False, clip_on=False,linewidth=0.5))

        for P in self.pockets:
            ax.add_artist(plt.Circle(P, 6, color='k', fill=True, clip_on=False,linewidth=0.5,alpha=0.5))

        ax.imshow(self.img, extent=[0, self.pool_table_size[0], 0, self.pool_table_size[1]], cmap='gray')

        #Use adjustable='box-forced' to make the plot area square-shaped as well.
        ax.set_aspect('equal', adjustable='box')

        return ax

    def draw_region_of_interest(self,ax,row_df):
        C=(row_df['C_x'], row_df['C_y'])
        T=(row_df['T_x'], row_df['T_y'])
        slope1=row_df['slope_X1T']
        slope2=row_df['slope_X2T']
        intercept1=row_df['intercept_X1T']
        intercept2=row_df['intercept_X2T']

        # we parametrize the line CT equation and compute the point 
        # that is 1000 distance from T
        t=1000
        far_away_point_x=C[0]+(T[0]-C[0])*t
        far_away_point_y=C[1]+(T[1]-C[1])*t
        far_away_point=(far_away_point_x,far_away_point_y)

        #line CT 
        slope=(C[1]-T[1])/(C[0]-T[0])
        intercept=T[1]-slope*T[0]

        #line perpendicular to CT that contains far_away_point
        slope_perpendicular=-1/slope
        intercept_perpendicular=far_away_point[1]-slope_perpendicular*far_away_point[0]

        #intersection between two lines
        x1=(-intercept1+intercept_perpendicular)/(-slope_perpendicular+slope1)
        x2=(-intercept2+intercept_perpendicular)/(-slope_perpendicular+slope2)
        y1=(slope1*intercept_perpendicular-slope_perpendicular*intercept1)/(-slope_perpendicular+slope1)
        y2=(slope2*intercept_perpendicular-slope_perpendicular*intercept2)/(-slope_perpendicular+slope2)
        
        points = np.array([[T[0],T[1]], [x1,y1], [x2,y2]])

        ax.add_patch(plt.Polygon(points, color='g',fill=True, clip_on=True,alpha=0.2))

        return ax

    def draw_point(ax,point):
        ax.add_artist(plt.Circle(point, 0.1, color='orange',fill=True, clip_on=False,linewidth=0.5))
        #Use adjustable='box-forced' to make the plot area square-shaped as well.
        ax.set_aspect('equal', adjustable='box')

        return ax

    def draw_specific_configuration(self,ax,row_df):
        C_estimated=(row_df['C_estimated_x'], row_df['C_estimated_y'])
        T_estimated=(row_df['T_estimated_x'], row_df['T_estimated_y'])
        ax=self.draw_pool_table_with_pockets(ax)
        ax=self.draw_ideal_configuration(ax,row_df)
        ax=self.draw_real_configuration(ax,row_df)
        ax=self.draw_point(ax,C_estimated)
        ax=self.draw_point(ax,T_estimated)
        return ax
