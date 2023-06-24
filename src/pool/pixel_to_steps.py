import os
import numpy as np
from pool.utils import Params

class PixelSteps:

    def __init__(self):
        self.params=Params()

    def load_calibration_points(self):
        return np.load(os.path.join(self.params.PATH_REPO, 'data', 'calibration_points.npy'))

    def generate_and_save_calibration_points(self, size):
        num_horizontal_points, num_vertical_points = size
        alpha = 1/(num_horizontal_points+1)
        beta = 1/(num_vertical_points+1)
        points = np.mgrid[1:num_horizontal_points+1,1:num_vertical_points+1].T.reshape(-1,2)
        points = points.astype(np.float32)
        rescaled_points = np.zeros_like(points)
        rescaled_points[:,0] = alpha*points[:,0]
        rescaled_points[:,1] = beta*points[:,1]
        np.random.shuffle(rescaled_points) 
        np.save(os.path.join(self.params.PATH_REPO, 'data', 'calibration_points.npy'), points)

        return rescaled_points
    
    def add_homing_position(self,points):
        home_point = [0,0]
        points = np.vstack([home_point,points]) 
        return points

    def cm_to_steps(self, incr_x, incr_y):
        scaler=self.params.CM_TO_STEPS
        W,H=self.params.GRID_SIZE_CM 
        phi1 = 1/(2*scaler) * (-incr_x*W+incr_y*H)
        phi2 = 1/(2*scaler) * (incr_x*W+incr_y*H)
        return phi1,phi2
    
    def cm_to_steps_vectorized(self, incr_x, incr_y):
        scaler=self.params.CM_TO_STEPS
        W,H=self.params.GRID_SIZE_CM 
        phi1 = 1/(2*scaler) * (-incr_x*W+incr_y*H)
        phi2 = 1/(2*scaler) * (incr_x*W+incr_y*H)
        return phi1, phi2
    
    def points_to_dict(self,points):
        d_points={}
        for i,point in enumerate(points):
            d_points[i]=point
        return d_points
    
    def img_num_to_incr_id(self,img_num):
        if img_num==0:
            return np.nan
        else:
            return f'{img_num-1}-{img_num}'
    
