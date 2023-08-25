import cv2
import os
import numpy as np
from pool.utils import Params

class Eye(object):
    """
    A class that finds preprocesses images: undistortion and prespective transformations 
    """

    def __init__(self, 
                 cameraMatrix = None,
                 dist = None,
                 arucos_pool_frame = None):
        self.params=Params()
        if cameraMatrix is None:
            self.cameraMatrix = np.load(os.path.join(self.params.PATH_REPO, 'data', 'cameraMatrix.npy'))
        else:
            self.cameraMatrix = cameraMatrix
        if dist is None:
            self.dist = np.load(os.path.join(self.params.PATH_REPO, 'data', 'dist.npy'))
        else:
            self.dist = dist
        if arucos_pool_frame is None:
            self.arucos_pool_frame = self.params.ARUCOS_POOL_FRAME
        else:
            self.arucos_pool_frame = arucos_pool_frame
        
    @staticmethod
    def _intersect(line1,line2): # TODO move to utils?
        vx1,vy1,x1,y1=line1
        vx2,vy2,x2,y2=line2
        t = (vy2*(x2-x1)-vx2*(y2-y1))/(vx1*vy2-vx2*vy1)
        return (x1+vx1*t,y1+vy1*t)

    def undistort_image(self,img, remapping=False):
        h,  w = img.shape[:2]
        # this new matrix is only for when you don't want to the black sides while undistorting an entire image
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.dist, (w,h), 1, (w,h))

        if not remapping:
            # Undistort
            dst = cv2.undistort(img, self.cameraMatrix, self.dist, None, newCameraMatrix)
            x, y, w, h = roi
            dst_cropped = dst[y:y+h, x:x+w]

        else:
            # Undistort with Remapping
            mapx, mapy = cv2.initUndistortRectifyMap(self.cameraMatrix, self.dist, None, newCameraMatrix, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # crop the image
            x, y, w, h = roi
            dst_cropped = dst[y:y+h, x:x+w]
        
        return dst_cropped

    def undistort_and_warp_image(self, img, img_corners=None):
        if img_corners is None:
            img_corners=cv2.imread(os.path.join(self.params.PATH_REPO, 'data', 'corners_0.jpg'))
        img_corners_undist=self.undistort_image(img_corners, remapping=False)
        undist_corners=self.get_pool_corners(img_corners_undist)
        undist_matrix=self.calculate_perspective_matrix(undist_corners)
        img_undist=self.undistort_image(img, remapping=False)
        img_undist_warp=self.transform_image_given_a_matrix(img_undist, undist_corners, undist_matrix)
        return img_undist_warp

    def get_pool_corners(self, img):
        top_aruco_ids, right_aruco_ids, bottom_aruco_ids, left_aruco_ids = self.arucos_pool_frame
        id_to_centroids = self.find_all_aruco_coordinates(img)
        bottomLine=np.array([id_to_centroids[aruco_id] for aruco_id in bottom_aruco_ids])
        topLine=   np.array([id_to_centroids[aruco_id] for aruco_id in top_aruco_ids])
        leftLine= np.array([id_to_centroids[aruco_id] for aruco_id in left_aruco_ids])
        rightLine=  np.array([id_to_centroids[aruco_id] for aruco_id in right_aruco_ids])
        lines={}

        for edge,position in zip([bottomLine,topLine,rightLine,leftLine], ['bottom', 'top', 'right', 'left']):
            # apply fitline() function
            [vx,vy,x,y] = cv2.fitLine(edge,cv2.DIST_L2,0,0.01,0.01)
            lines[position]=[vx,vy,x,y] 
        
        #save corner in the following order
        #0 - top-left
        #1 - top-right
        #2 - bottom-right
        #3 - bottom-left
        pool_corners=[]
        for pair_of_lines in [('top','left'),('top','right'),('bottom','right'), ('bottom','left')]:
            horizontal,vertical=pair_of_lines
            hline=lines[horizontal]
            vline=lines[vertical]
            cX,cY=self._intersect(hline,vline)
            corner=(int(cX), int(cY))
            pool_corners+=[corner]

        return pool_corners
    
    def find_all_aruco_coordinates(self, img, debug_path='./results/', debug=False):
        arucoDict=cv2.aruco.DICT_4X4_100
        arucoDict = cv2.aruco.getPredefinedDictionary(arucoDict)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoDetector = cv2.aruco.ArucoDetector(
            arucoDict, arucoParams)

        corners, ids, rejected = arucoDetector.detectMarkers(img)
        id_to_centroids={}
        if debug:
            img_to_draw=img.copy()

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = (topLeft[0] + bottomRight[0]) / 2.0
                cY = (topLeft[1] + bottomRight[1]) / 2.0
                id_to_centroids[markerID]=(cX,cY)

                if debug:
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # draw the bounding box of the ArUCo detection
                    cv2.line(img_to_draw, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(img_to_draw, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(img_to_draw, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(img_to_draw, bottomLeft, topLeft, (0, 255, 0), 2)
                    cv2.circle(img_to_draw, (int(cX), int(cY)), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the img
                    cv2.putText(img_to_draw, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (0, 255, 0), 2)
        if debug:
            cv2.imwrite(f'{debug_path}all_arucos_detected.png', img_to_draw)
        return id_to_centroids
    
    def get_aruco_coordinates_given_aruco_id(self, img, aruco_to_track):
        id_to_centroids=self.find_all_aruco_coordinates(img)
        if aruco_to_track in id_to_centroids: 
            return id_to_centroids[aruco_to_track]
        else:   
            raise ValueError('the aruco specified is not found in the img')
    
    def get_aruco_coordinates_given_several_aruco_ids(self, img, arucos_to_track):
        id_to_centroids=self.find_all_aruco_coordinates(img)
        return {k: v for k, v in id_to_centroids.items() if k in arucos_to_track}

    def calculate_perspective_matrix(self, corners) -> np.ndarray:

        # Order points in clockwise order:
        # First, we separate corners into individual points
        # Index 0 - top-left
        #       1 - top-right
        #       2 - bottom-right
        #       3 - bottom-left
        top_l, top_r, bottom_r, bottom_l = corners[0], corners[1], corners[2], corners[3]

        # Determine width of new image which is the max distance between 
        # (bottom right and bottom left) or (top right and top left) x-coordinates
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))

        # Determine height of new image which is the max distance between 
        # (top right and bottom right) or (top left and bottom left) y-coordinates
        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))

        # Construct new points to obtain top-down view of image in 
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                        [0, height - 1]], dtype = "float32")

        # Convert to Numpy format
        corners = np.array(corners, dtype="float32")

        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners, dimensions)

        return matrix
    
    def transform_image_given_a_matrix(self, image, corners, matrix):
        # Return the transformed image

        # width  = image.shape[1]
        # height = image.shape[0]

        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]

        # Determine width of new image which is the max distance between 
        # (bottom right and bottom left) or (top right and top left) x-coordinates
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))

        # Determine height of new image which is the max distance between 
        # (top right and bottom right) or (top left and bottom left) y-coordinates
        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))
        
        return cv2.warpPerspective(image, matrix, (width, height))

    def transform_point_given_a_matrix(self, point, matrix):
        x,y = point
        point_to_transform = np.array([[x,y]], dtype='float32')
        point_to_transform = np.array([point_to_transform])
        transformed_point = cv2.perspectiveTransform(point_to_transform, matrix)
        x_transformed, y_transformed = transformed_point[0][0]
        return x_transformed, y_transformed