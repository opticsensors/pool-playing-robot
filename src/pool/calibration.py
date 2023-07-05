import os
import cv2
import numpy as np
from pool.utils import Params
from pool.eye import Eye

class InverseKinematics:

    def __init__(self):
        self.params=Params()

    def generate_calibration_points(self, size, reduce=(1,1), random_order=True):
        num_horizontal_points, num_vertical_points = size
        alpha = 1/(num_horizontal_points+1)
        beta = 1/(num_vertical_points+1)
        points = np.mgrid[1:num_horizontal_points+1,1:num_vertical_points+1].T.reshape(-1,2)
        points = points.astype(np.float32)
        points[:,0] = alpha*points[:,0]
        points[:,1] = beta*points[:,1]
        minx,miny=points[0,:]
        maxx,maxy=points[-1,:]
        midx=(minx+maxx)/2
        midy=(miny+maxy)/2
        redux, reduy = reduce
        points[:,0]=(points[:,0]-midx)*redux+midx
        points[:,1]=(points[:,1]-midy)*reduy+midy
        if random_order:
            np.random.shuffle(points) 
        return points

    def cm_to_steps(self, incr_x, incr_y):
        scaler=self.params.CM_TO_STEPS
        W,H=self.params.GRID_SIZE_CM 
        phi1 = 1/(2*scaler) * (-incr_x*W+incr_y*H)
        phi2 = 1/(2*scaler) * (incr_x*W+incr_y*H)
        return phi1,phi2
    
    def img_num_to_incr_id(self,img_num):
        if img_num==0:
            return np.nan
        else:
            return f'{img_num-1}-{img_num}'
    
class CameraCalibration:
    
    def __init__(self,
                 cameraMatrix = None,
                 ):
        self.eye = Eye()
        self.params = Params()
        if cameraMatrix is None:
            self.cameraMatrix = self.eye.cameraMatrix
        else:
            self.cameraMatrix = cameraMatrix
        self.dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) #input images should always be undistorted, so pinhole camera model can be applied !

    def find_chessboard_corners(self, img, chessboardSize, distorted=False):
        if distorted:
            img = self.eye.undistort_image(img, remapping=False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) -> 3d points in real world space
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgp=corners2.reshape(-1,2) # 2d points in image plane (same shape as objp)
            return objp, imgp
        else:
            raise ValueError('the chessboard is not found in the img')
    
    def draw_chess_board_corners(self, img, chessboardSize, distorted=False):
        if distorted:
            img = self.eye.undistort_image(img, remapping=False)
        img_to_draw=img.copy()
        objp, imgp = self.find_chessboard_corners( img, chessboardSize)
        # Draw and display the corners
        img_to_draw=cv2.drawChessboardCorners(img_to_draw, chessboardSize, imgp.reshape(-1,1,2), True)
        return img_to_draw
    
    def generate_and_save_world_frame(self, img, chessboardSize):
        objp, imgp = self.find_chessboard_corners( img, chessboardSize)
        found,rvec,tvec = cv2.solvePnP(objp, imgp, self.cameraMatrix, self.dist)
        np.save(os.path.join(self.params.PATH_REPO,'data','rvec.npy'), rvec)
        np.save(os.path.join(self.params.PATH_REPO,'data','tvec.npy'), tvec)
        return rvec,tvec
    
    def load_world_frame(self):
        rvec = np.load(os.path.join(self.params.PATH_REPO, 'data', 'rvec.npy'))
        tvec = np.load(os.path.join(self.params.PATH_REPO, 'data', 'tvec.npy'))
        return rvec, tvec

    def transform_image_point_to_world_point(self, uvPoint, z_plane ,rvec=None, tvec=None, distorted=False):
        """
        Find 3D coordinates expressed in the world frame that are located
        in a known plane (ball centroid plane) using 2D coordinates from image plane
        """
        assert uvPoint.shape == (3,1) and uvPoint[2,0] == 1
        if distorted:
            img = self.eye.undistort_image(img, remapping=False)
        if rvec is None and tvec is None:
            rvec,tvec=self.load_world_frame()
        rotMatrix = cv2.Rodrigues(rvec)[0]
        rotMatrix_inv = np.linalg.inv(rotMatrix)
        cameraMatrix_inv = np.linalg.inv(self.cameraMatrix) 
        left_vector=rotMatrix_inv@cameraMatrix_inv@uvPoint
        right_vector=rotMatrix_inv@tvec
        s=(z_plane+right_vector[2,0])/(left_vector[2,0])
        xyzPoint=rotMatrix_inv@(s*cameraMatrix_inv@uvPoint-tvec)
        return xyzPoint.ravel()
    
    def transform_world_point_to_img_point(self, xyzPoint, rvec=None, tvec=None, distorted=False):
        """
        Find 2D coordinates in image plane given 3D coordinates 
        expressed in the world frame using opencv function
        """
        if distorted:
            img = self.eye.undistort_image(img, remapping=False)
        if rvec is None and tvec is None:
            rvec,tvec=self.load_world_frame()
        uvPoint=cv2.projectPoints(xyzPoint, rvec, tvec, self.cameraMatrix, self.dist)[0]
        return uvPoint.ravel()
    
    def predict_carriage_position_in_image_plane(self, ball_centroid, angle, offset, rvec=None, tvec=None):
        if rvec is None and tvec is None:
            rvec,tvec=self.load_world_frame()
        uvBallCentroid=np.pad(ball_centroid, (0, 1), constant_values=1).reshape(-1,1)
        trans=self.transform_image_point_to_world_point(uvBallCentroid, 0, rvec, tvec) #translation vector from world frame to auxiliar frame 
        angle=angle-90 # conversion angle uv frame to angle auxiliar frame (the frame with x axis with the same direction as flipper)
        angle=np.radians(angle)
        Rt=np.array([[np.cos(angle), -np.sin(angle), 0, trans[0]],
                    [np.sin(angle), np.cos(angle), 0, trans[1]],
                    [0, 0, 1, trans[2]],
                    [0, 0, 0, 1]])
        a,b,d,h=offset
        offset_point_of_contact=np.array([-b,-a,-h,1]).reshape(-1,1)
        offset_rotation_axis=np.array([0,-d,0,1]).reshape(-1,1)
        carriage_aruco_point3D = (Rt@offset_point_of_contact)[:3,0] + offset_rotation_axis[:3,0]
        carriage_aruco_point2D = self.transform_world_point_to_img_point(carriage_aruco_point3D ,rvec, tvec)
        return carriage_aruco_point2D

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
        sorted_aruco_coord = dict(sorted(d_aruco_coord.items(), reverse=True)) # flipper direction goes from higher aruco values to lower 
        line_points=np.array(list(sorted_aruco_coord.values()))
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
    
    def get_flipper_angle(self, img, flipper_arucos):
        flipper_dist=self.eye.get_aruco_coordinates_given_several_aruco_ids(img,flipper_arucos)
        angle=self.get_angle_given_dict_of_aruco_coord(flipper_dist)
        return angle
    
    def draw_flipper_angle(self, img, flipper_arucos):
        flipper_dist=self.eye.get_aruco_coordinates_given_several_aruco_ids(img,flipper_arucos)
        line_points=np.array(list(flipper_dist.values()))
        [vx,vy,x,y] = cv2.fitLine(line_points,cv2.DIST_L2,0,0.01,0.01)
        # Now find two extreme points on the line to draw line
        lefty = int((-x*vy/vx) + y)
        righty = int(((img.shape[1]-x)*vy/vx)+y)
        #Finally draw the line
        img_to_draw=img.copy()
        cv2.line(img_to_draw,(img_to_draw.shape[1]-1,righty),(0,lefty),(0,255,0),3)
        return img_to_draw