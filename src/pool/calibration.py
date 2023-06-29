import os
import cv2
import numpy as np
from pool.utils import Params
from pool.eye import Eye

class PixelSteps:

    def __init__(self):
        self.params=Params()

    def load_calibration_points(self):
        return np.load(os.path.join(self.params.PATH_REPO, 'data', 'calibration_points.npy')) #TODO change name to stepper_calibration_points

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
        np.save(os.path.join(self.params.PATH_REPO, 'data', 'calibration_points.npy'), points) #TODO change name to stepper_calibration_points

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
    
class BallCarriage:
    
    def __init__(self):
        self.params=Params()

    def load_calibration_points(self):
        return np.load(os.path.join(self.params.PATH_REPO, 'data', 'system_calibration_points.npy'))
    

class DataExtractor:
    def __init__(self,
                 img_corners=None):
        self.params=Params()
        self.eye=Eye()
        if img_corners is None:
            img_corners=cv2.imread(os.path.join(self.params.PATH_REPO, 'data', 'corners_0.jpg'))
        #compute corners 
        undist_img=self.eye.undistort_image(img_corners, remapping=False)
        dist_corners=self.eye.get_pool_corners(img_corners)
        undist_corners=self.eye.get_pool_corners(undist_img)

        #compute prespective transform matrix
        # the prespective trans matrix should be the same for all the captured images
        self.dist_matrix=self.eye.calculate_perspective_matrix(dist_corners)
        self.undist_matrix=self.eye.calculate_perspective_matrix(undist_corners)

    def get_single_aruco_data(self, img, aruco):
        undistorted=self.eye.undistort_image(img, remapping=False)
        coord_dist=self.eye.get_aruco_coordinates_given_aruco_id(img, aruco)
        coord_undist=self.eye.get_aruco_coordinates_given_aruco_id(undistorted, aruco)
        coord_dist_warp = self.eye.transform_point_given_a_matrix(coord_dist,self.dist_matrix)
        coord_undist_warp = self.eye.transform_point_given_a_matrix(coord_undist,self.undist_matrix)
        coord={
            'dist':coord_dist,
            'undist': coord_undist,
            'dist_warp':coord_dist_warp,
            'undist_warp':coord_undist_warp,
        }
        return coord
    
    def get_several_aruco_data(self, img, arucos):
        undistorted=self.eye.undistort_image(img, remapping=False)
        d_dist=self.eye.get_aruco_coordinates_given_several_aruco_ids(img,arucos)
        d_undist=self.eye.get_aruco_coordinates_given_several_aruco_ids(undistorted,arucos)
        d_dist_warp = {k:self.eye.transform_point_given_a_matrix(v,self.dist_matrix) for k,v in d_dist.items()}
        d_undist_warp = {k:self.eye.transform_point_given_a_matrix(v,self.undist_matrix) for k,v in d_undist.items()}
        d_coord={
            'd_dist':d_dist,
            'd_undist': d_undist,
            'd_dist_warp':d_dist_warp,
            'd_undist_warp':d_undist_warp,
        }
        return d_coord
    
    def get_angle_given_dict_of_aruco_coord(self,d_aruco_coord):
        line_points=np.array(list(d_aruco_coord.values()))
        if len(line_points)>1:
            #average vector angles
            degrees=[]
            indices=np.transpose(np.triu_indices(line_points.shape[0],1))
            for (strat_id, end_id) in (indices):
                start=line_points[strat_id,:]
                end=line_points[end_id,:]
                vector=end-start
                degrees.append(np.degrees(np.arctan2(vector[1],vector[0])))
            angle=np.mean(degrees)
        else:
            angle=np.nan
        return angle

    def get_angle_given_line_of_arucos(self, img, line_of_arucos):
        d_coord=self.get_several_aruco_data(img,line_of_arucos)
        d_angles={
            'angle_dist': self.get_angle_given_dict_of_aruco_coord(d_coord['d_dist']),
            'angle_undist': self.get_angle_given_dict_of_aruco_coord(d_coord['d_undist']),
            'angle_dist_warp': self.get_angle_given_dict_of_aruco_coord(d_coord['d_dist_warp']),
            'angle_undist_warp': self.get_angle_given_dict_of_aruco_coord(d_coord['d_undist_warp']),
        }
        return d_angles
    
    def get_coord_given_line_of_arucos(self, img, line_of_arucos):
        d_coord=self.get_several_aruco_data(img,line_of_arucos)
        d_flipper={
            'flipper_dist': np.array(list(d_coord['d_dist'].values())).reshape(1,-1),
            'flipper_undist': np.array(list(d_coord['d_undist'].values())).reshape(1,-1),
            'flipper_dist_warp': np.array(list(d_coord['d_dist_warp'].values())).reshape(1,-1),
            'flipper_undist_warp': np.array(list(d_coord['d_undist_warp'].values())).reshape(1,-1),
        }
        return d_flipper
    
    def debug_flipper_line(self, img, arucos):
        d_dist=self.eye.get_aruco_coordinates_given_several_aruco_ids(img,arucos)
        line_points=np.array(list(d_dist.values()))
        [vx,vy,x,y] = cv2.fitLine(line_points,cv2.DIST_L2,0,0.01,0.01)
        print(vx,vy,x,y)
        # Now find two extreme points on the line to draw line
        lefty = int((-x*vy/vx) + y)
        righty = int(((img.shape[1]-x)*vy/vx)+y)
        #Finally draw the line
        cv2.line(img,(img.shape[1]-1,righty),(0,lefty),(0,255,0),3)
        return img