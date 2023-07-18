import numpy as np
import pandas as pd
from pool.utils import Params
from pool.brain import CTP,CBTP,CTTP,CTBP
from pool.pool_env import PoolEnv
from pool.pool_frame import PoolFrame

class BruteForce:
    def __init__(self):
        self.params=Params()

    def angle_sweep(self, env, config, turn, angles_to_study):
        dict_to_save = {}
        list_of_dict = []
        for action in angles_to_study:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                dict_to_save['turn']=info['turn']
                dict_to_save['total_collisions']=info['total_collisions']
                dict_to_save['first_ball_collision']=info['first_ball_collision']
                if info['potted_ball'] == 'correct_ball':
                    dict_to_save['potted_ball'] = 1
                else:
                    dict_to_save['potted_ball'] = 0
                dict_to_save['action']=info['action']

                list_of_dict.append(dict_to_save.copy())
                env.close()
                env=PoolEnv(render_mode = None)
                observation, info = env.reset(config, turn)

        df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
        env.close()
        return df
    
    def find_easiest_shot(self, df):        
        data=df['potted_ball'].to_list()
        centroids=[]
        weights=[]
        prev_val=0
        for i,value in enumerate(data):
            #from 0 to 1
            if value>prev_val:
                prev_val=value
                i_ini=i
            #from 1 to 0
            elif value < prev_val:
                l=i-i_ini
                centroid=(i+i_ini)//2
                weights.append(l)
                centroids.append(centroid)
                prev_val=value
        arr_cetroids=np.array(centroids)
        arr_weights=np.array(weights)
        try:
            maximum=arr_cetroids[np.unravel_index(np.nanargmax(arr_weights, axis=None), arr_weights.shape)]
            angle=df['action'].iloc[maximum]
        except ValueError:
            angle=None
        return angle

    def map_angle(self, angle): 
        """
        maps angle from 0,360 to -180,180
        """
        if angle is not None:
            return (angle+180)%360-180
        else:
            return angle

    def get_actuator_angle(self, config, turn, first_sweep_precision=2, second_sweep_precision=0.4):
        if turn == 'solid':
            turn=0
        elif turn == 'strip':
            turn=1
        env=PoolEnv(render_mode = None)
        observation, info = env.reset(config, turn)
        first_sweep = list(np.arange(0, 360, first_sweep_precision))
        df1 = self.angle_sweep(env, config, turn, first_sweep)
        print('first sweep done!')
        possible_valid_actions = df1['action'][df1['first_ball_collision']==turn]
        possible_valid_ranges = []

        for action in possible_valid_actions:
            action_range = (action-second_sweep_precision, action+second_sweep_precision)
            possible_valid_ranges.append(action_range)

        result = []
        for item in sorted(possible_valid_ranges):
            result = result or [item]
            if item[0] > result[-1][1]:
                result.append(item)
            else:
                old = result[-1]
                result[-1] = (old[0], max(old[1], item[1]))

        second_sweep = []
        for angle_range in result:
            second_sweep.extend(list(np.arange(angle_range[0], angle_range[1],0.1)))
        df2 = self.angle_sweep(env, config, turn, second_sweep)
        print('second sweep done!')
        return self.map_angle(self.find_easiest_shot(df2))
    
    def debug(self, config, turn, angle):
        if turn == 'solid':
            turn=0
        elif turn == 'strip':
            turn=1
        env=PoolEnv(render_mode = 'human')
        observation, info = env.reset(config, turn)
        observation, reward, terminated, truncated, info = env.step(angle)
        env.close()

class GeomericSolution:
    def __init__(self, pool_frame=None, ball_radius=None):
        
        params = Params()
        if ball_radius is None:
            self.ball_radius=params.BALL_RADIUS
        else:
            self.ball_radius=ball_radius
        if pool_frame is None:
            self.pool_frame = PoolFrame()
        else:
            self.pool_frame = pool_frame
        
        self.ctp=CTP(self.pool_frame, self.ball_radius)
        self.cbtp=CBTP(self.pool_frame, self.ball_radius)
        self.cttp=CTTP(self.pool_frame, self.ball_radius)
        self.ctbp=CTBP(self.pool_frame, self.ball_radius)
    
    def correct_centroids(self, d_centroids, epsilon=0.1):
        new_d_centroids={}
        for ball_num, centroid in d_centroids.items():
            cx, cy = centroid
            if cx>self.pool_frame.right_x:
                cx=self.pool_frame.right_x+epsilon
            elif cx<self.pool_frame.left_x:
                cx=self.pool_frame.left_x+epsilon
            if cy>self.pool_frame.bottom_y:
                cy=self.pool_frame.bottom_y+epsilon
            elif cy<self.pool_frame.top_y:
                cy=self.pool_frame.top_y+epsilon
            new_d_centroids[ball_num]=[cx, cy]
        return new_d_centroids

    def get_actuator_angle(self, d_centroids, turn, shot_type=None):
        d_centroids=self.correct_centroids(d_centroids)
        df_ctp =self.ctp.selected_shots(d_centroids, turn)
        df_cbtp=self.cbtp.selected_shots(d_centroids, turn)
        df_cttp=self.cttp.selected_shots(d_centroids, turn)
        df_ctbp=self.ctbp.selected_shots(d_centroids, turn)
        angle=None

        if shot_type is None:
            if len(df_ctp)!=0:
                angle = df_ctp.iloc[0]['angle']
            elif len(df_cbtp)!=0:
                angle = df_cbtp.iloc[0]['angle']
            elif len(df_cttp)!=0:
                angle = df_cttp.iloc[0]['angle']
            elif len(df_ctbp)!=0:
                angle = df_ctbp.iloc[0]['angle']
        else:
            if shot_type=='CTP':
                angle = df_ctp.iloc[0]['angle']
            elif shot_type=='CBTP':
                angle = df_cbtp.iloc[0]['angle']
            elif shot_type=='CTTP':
                angle = df_cttp.iloc[0]['angle']
            elif shot_type=='CTBP':
                angle = df_ctbp.iloc[0]['angle']

        return angle

    def debug(self, img, d_centroids, turn, shot_type):
        d_centroids=self.correct_centroids(d_centroids)
        img_to_draw=img.copy()
        if shot_type == 'CTP':
            df =self.ctp.selected_shots(d_centroids, turn)   
            img=self.ctp.draw_all_trajectories(df, img_to_draw)
        elif shot_type == 'CBTP':
            df=self.cbtp.selected_shots(d_centroids, turn)
            img=self.cbtp.draw_all_trajectories(df, img_to_draw)
        elif shot_type == 'CTTP':
            df=self.cttp.selected_shots(d_centroids, turn)
            img=self.cttp.draw_all_trajectories(df, img_to_draw)
        elif shot_type == 'CTBP':
            df=self.ctbp.selected_shots(d_centroids, turn)
            img=self.ctbp.draw_all_trajectories(df, img_to_draw)
        img_to_draw=self.pool_frame.draw_pool_balls(img_to_draw, d_centroids)
        img_to_draw=self.pool_frame.draw_frame(img_to_draw)
        return img_to_draw, df