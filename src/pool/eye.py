import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

class Eye(object):
    """
    A class that finds the centroid of the pool balls and classifies them

    Attributes
    ----------    
    ...

    """

    #we map each color with a number:
    NUM_TO_COLOR={
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
    COLOR_AND_TYPE_TO_NUM={
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
    COLOR_TO_LAB={
        'white'   : [227, 130, 148],
        'yellow'  : [216.18407452, 124.80355131, 195.73519079],
        'blue'    : [89.64986175, 134.93342045, 103.84520826],
        'red'     : [147.45829885, 172.49603901, 178.22393935],
        'purple'  : [78.6311673, 135.5586135, 123.057679],
        'orange'  : [180.62554741, 151.8795586 , 187.98600759],
        'green'   : [110.05696026, 106.50088757, 132.41375192],
        'burgundy': [121.56706638, 160.19773476, 158.41379201],
        'black'   : [58, 129, 136]
    }
    
    WHITE_LOWER_LAB=[175, 0, 0]
    WHITE_UPPER_LAB=[255, 147, 164]
    WHITE_LOWER_HSV=[0, 0, 179]
    WHITE_UPPER_HSV=[180, 106, 255]
    
    RECTANGLE_AREA=382352 #HxV
    BALL_AREA=1134.115 #PI*RADI^2
    RATIO_BALL_RECTANGLE=0.00296615132

    def __init__(self, **kwargs):
        param = {  # with defaults
            'lower_lab': Eye.WHITE_LOWER_LAB,
            'upper_lab': Eye.WHITE_UPPER_LAB,
            'lower_hsv': Eye.WHITE_LOWER_HSV,
            'upper_hsv': Eye.WHITE_UPPER_HSV,
            'colors':    Eye.COLOR_TO_LAB,
            }
        param.update(kwargs)
        
        self._lower_hsv=param['lower_hsv']
        self._lower_lab=param['lower_lab']
        self._upper_hsv=param['upper_hsv']
        self._upper_lab=param['upper_lab']
        self._color_to_lab=param['colors']

    @staticmethod
    def _dict_to_arr( color_to_lab):

        color_to_num = {v: k for k, v in Eye.NUM_TO_COLOR.items()}
        num_to_lab={color_to_num[k]: v for k, v in color_to_lab.items()}
        lab_arr=np.zeros((len(color_to_lab),3))
        for num in num_to_lab:
            lab=num_to_lab[num]
            lab_arr[num,:]=np.array(lab)
        
        return lab_arr

    @staticmethod
    def _intersect(line1,line2):
        vx1,vy1,x1,y1=line1
        vx2,vy2,x2,y2=line2
        t = (vy2*(x2-x1)-vx2*(y2-y1))/(vx1*vy2-vx2*vy1)
        return (x1+vx1*t,y1+vy1*t)

    @property
    def lower_hsv(self):
        return np.array(self._lower_hsv)
    
    @lower_hsv.setter
    def lower_hsv(self,color):
        if isinstance(color, list):
            self._lower_hsv=color

    @property
    def lower_lab(self):
        return np.array(self._lower_lab)
    
    @lower_lab.setter
    def lower_lab(self,color):
        if isinstance(color, list):
            self._lower_lab=color

    @property
    def upper_hsv(self):
        return np.array(self._upper_hsv)
    
    @upper_hsv.setter
    def upper_hsv(self,color):
        if isinstance(color, list):
            self._upper_hsv=color

    @property
    def upper_lab(self):
        return np.array(self._upper_lab)
    
    @upper_lab.setter
    def upper_lab(self,color):
        if isinstance(color, list):
            self._upper_lab=color

    @property
    def color_to_lab(self):
        lab_arr=self._dict_to_arr(self._color_to_lab)
        return lab_arr
    
    @color_to_lab.setter
    def color_to_lab(self,color_to_lab):
        if isinstance(color_to_lab, dict):
            self._color_to_lab=color_to_lab

    def undistort_image(self,img, cameraMatrix, dist, remapping=False):
        h,  w = img.shape[:2]
        # this new matrix is only for when you don't want to the black sides while undistorting an entire image
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

        if not remapping:
            # Undistort
            dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
            x, y, w, h = roi
            dst_cropped = dst[y:y+h, x:x+w]

        else:
            # Undistort with Remapping
            mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # crop the image
            x, y, w, h = roi
            dst_cropped = dst[y:y+h, x:x+w]
        
        return dst_cropped

    def get_pool_corners(self, img, left_aruco_ids, right_aruco_ids, top_aruco_ids, bottom_aruco_ids):

        arucoDict=cv2.aruco.DICT_4X4_100
        arucoDict = cv2.aruco.getPredefinedDictionary(arucoDict)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoDetector = cv2.aruco.ArucoDetector(
            arucoDict, arucoParams)

        corners, ids, rejected = arucoDetector.detectMarkers(img)
        id_to_centroids={}

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
                #print(f"[INFO] ArUco marker ID: {markerID}, ({cX},{cY})")
        bottomLine=np.array([id_to_centroids[aruco_id] for aruco_id in bottom_aruco_ids])
        topLine=   np.array([id_to_centroids[aruco_id] for aruco_id in top_aruco_ids])
        rightLine= np.array([id_to_centroids[aruco_id] for aruco_id in left_aruco_ids])
        leftLine=  np.array([id_to_centroids[aruco_id] for aruco_id in right_aruco_ids])
        lines={}

        for edge,position in zip([bottomLine,topLine,rightLine,leftLine], ['bottom', 'top', 'right', 'left']):
            # apply fitline() function
            [vx,vy,x,y] = cv2.fitLine(edge,cv2.DIST_L2,0,0.01,0.01)
            lines[position]=[vx,vy,x,y] 
        
        #save corner in the following order
        #0 - top-right
        #1 - top-left
        #2 - bottom-left
        #3 - bottom-right
        pool_corners=[]
        for pair_of_lines in [('top','right'),('top','left'),('bottom','left'), ('bottom','right')]:
            horizontal,vertical=pair_of_lines
            hline=lines[horizontal]
            vline=lines[vertical]
            cX,cY=self._intersect(hline,vline)
            corner=(int(cX), int(cY))
            pool_corners+=[corner]

        return pool_corners
    
    def get_aruco_coordinates(self, img, aruco_to_track):

        arucoDict=cv2.aruco.DICT_4X4_100
        arucoDict = cv2.aruco.getPredefinedDictionary(arucoDict)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoDetector = cv2.aruco.ArucoDetector(
            arucoDict, arucoParams)

        corners, ids, rejected = arucoDetector.detectMarkers(img)
        id_to_centroids={}

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
        
        if aruco_to_track in id_to_centroids:
            return id_to_centroids[aruco_to_track]
        else:   
            raise ValueError('the aruco specified is not found in the img')
            

    def get_cloth_color(self,hsv,search_width=45):
        """
        Find the most common HSV values in the image.
        In a well lit image, this will be the cloth
        """

        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]) # origina: hist = cv2.calcHist([hsv], [1], None, [180], [0, 180])
        h_max = np.unravel_index(np.nanargmax(hist, axis=None), hist.shape)[0]
        
        hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        s_max = np.unravel_index(np.nanargmax(hist, axis=None), hist.shape)[0]
        
        hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        v_max = np.unravel_index(np.nanargmax(hist, axis=None), hist.shape)[0]

        # define range of blue color in HSV
        lower_color = np.array([h_max-search_width,s_max-search_width,v_max-search_width])
        upper_color = np.array([h_max+search_width,s_max+search_width,v_max+search_width])

        return lower_color, upper_color

    def color_segmentation(self,hsv, lower_color, upper_color,**kwargs):
        """
        Find the region where pixels are between lower_color and upper_color
        """

        param = {  # with defaults
            'filter_radius': 17,
            'kernel_size': 11,
            'iterations': 2,
            'postprocessing': 'median'
            }
        param.update(kwargs)

        # Threshold the HSV image to get only cloth colors
        mask = cv2.inRange(hsv, lower_color, upper_color)
        inverted_mask=cv2.bitwise_not(mask)

        if param['postprocessing']=='median':
            #use a median filter to get rid of speckle noise
            median = cv2.medianBlur(mask,param['filter_radius'])
            prostprocessed_mask = cv2.bitwise_not(median)

        elif param['postprocessing']=='morph':
            #use morphological operations to get rid of noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (param['kernel_size'],param['kernel_size']))
            prostprocessed_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_OPEN, kernel,iterations = param['iterations'])

        return inverted_mask,prostprocessed_mask

    def perspective_transform(self,image, corners):
        """
        4 point prespective transform.
        Aligns pool table edges with image edges.
        """

        # Order points in clockwise order:
        # First, we separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
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

        # Construct new points to obtain top-down view of image in 
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                        [0, height - 1]], dtype = "float32")

        # Convert to Numpy format
        corners = np.array(corners, dtype="float32")

        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners, dimensions)

        # Return the transformed image
        return cv2.warpPerspective(image, matrix, (width, height)), matrix

    def crop_image(self, img, h_offset, v_offset):
        return img[v_offset:-v_offset, h_offset:-h_offset]

    def substract_background(self, thresh, bg_mask):
        thresh[bg_mask==255]=0
        return thresh

    def remove_small_dots(self,thresh,connectivity):
        """
        Finds all image blobs that are below or between an area threshold and gets rid of them.
        Those dots aren't considered part of the pattern and they are classified as noise.
        
        Parameters
        ----------
        
        """
        #define max and min size
        factor_of_safety=0.15
        H=thresh.shape[1]
        V=thresh.shape[0]
        min_size=factor_of_safety*H*V*Eye.RATIO_BALL_RECTANGLE 
        #max_size=16*H*V*Eye.RATIO_BALL_RECTANGLE #all balls connected

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, 
                                                                                    connectivity=connectivity)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        
        #size of all found blobs (the first one is the background)
        sizes = stats[:, -1] 
        # we have to treat single and connected blobs differently:
        # we will save the binary image for interesting blobs
        blobs=np.zeros((output.shape), dtype=np.uint8)

        #we satrt at 1 because first element is bg
        for i in range(1, nb_components): 
            
            # check if blob is between desired sizes
            if sizes[i]>min_size:
                blobs[output==i] = 255

        return blobs


    def find_ball_blobs(self,thresh):

        d_centroids={}

        # Compute Euclidean distance from every binary pixel
        # to the nearest zero pixel then find peaks
        distance_map = ndimage.distance_transform_edt(thresh)
        local_max = peak_local_max(distance_map, indices=False, min_distance=20, labels=thresh)

        # Perform connected component analysis then apply Watershed
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance_map, markers, mask=thresh)
        #l_unique=[ball_num for ball_num in labels if ball_num!=0]

        for i in np.unique(labels):
            if i!=0:
                masked_ball = np.zeros((thresh.shape), dtype=np.uint8)#1 ch
                masked_ball[labels==i] = 255
                M = cv2.moments(masked_ball)
                # calculate x,y coordinate of center
                centroid= [(M["m10"] / M["m00"]),(M["m01"] / M["m00"])]
                d_centroids[i]=centroid

        return labels,d_centroids

    def tune_white_color(self,img,numbered_balls):
        """

        """
        for i in np.unique(numbered_balls):
            if i!=0:
                masked_ball = np.zeros((img.shape[:-1]), dtype=np.uint8)#1 ch
                masked_ball[numbered_balls==i] = 255
                masked = cv2.bitwise_and(img, img, mask=masked_ball)
                contour_ball,_ = cv2.findContours(masked_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contour_ball[0])
                masked=masked[y:y+h, x:x+w, :]
                cv2.imwrite(f'./results/masked_{i}.png', masked)

    def classify_balls(self,img,numbered_balls,d_centroids,color_space='hsv'):
        """
        For each labeled blob, we need to decide its color (white, yellow, ...) and its type (solid, striped)
        """

        #converts the default labeled number to the new classified number for each ball
        old_to_new={}

        for i in np.unique(numbered_balls):
            if i!=0:
                masked_ball = np.zeros((img.shape[:-1]), dtype=np.uint8)#1 ch
                masked_ball[numbered_balls==i] = 255
                masked = cv2.bitwise_and(img, img, mask=masked_ball)
                contour_ball,_ = cv2.findContours(masked_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contour_ball[0])
                masked_ball=masked_ball[y:y+h, x:x+w]
                masked=masked[y:y+h, x:x+w, :]

                masked_hsv=cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
                masked_lab=cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)

                if color_space=='hsv':
                    lower = self.lower_hsv
                    upper = self.upper_hsv
                    thresholded_ball = cv2.inRange(masked_hsv, lower, upper)
                elif color_space=='lab':
                    lower = self.lower_lab
                    upper = self.upper_lab
                    thresholded_ball = cv2.inRange(masked_lab, lower, upper)
                    
                
                num_pixels_ball=np.count_nonzero((numbered_balls==i))
                num_white_pixels_ball=np.count_nonzero(thresholded_ball)
                proportion_white_pixels=num_white_pixels_ball/num_pixels_ball

                if 0.13<=proportion_white_pixels<0.85:
                    ball_type='striped'
                elif proportion_white_pixels<0.13:
                    ball_type='solid'
                else:
                    ball_type='cue ball'

                if ball_type != 'cue ball':
                    avg_lab=masked_lab[(thresholded_ball==0) & (masked_ball==255)].mean(axis=0)
                    
                else:
                    avg_lab=masked_lab[(masked_ball==255)].mean(axis=0)

                measured_lab=np.tile(avg_lab, (9, 1))#there are 9 colors
                error=np.sqrt(np.sum(np.power(measured_lab[:,1:] - self.color_to_lab[:,1:], 2), axis=1)) #we use chroma instead of euclidean distance
                row_with_min_error=np.unravel_index(np.nanargmin(error, axis=None), error.shape)[0]
                ball_color=Eye.NUM_TO_COLOR[row_with_min_error]
                ball_number=Eye.COLOR_AND_TYPE_TO_NUM[ball_color][ball_type]
                old_to_new[i]=ball_number
                print(f'ball {i}= %white: {proportion_white_pixels}, color: {ball_color}, ball_type: {ball_type}, ball num: {ball_number}')

        return {old_to_new[k]: v for k, v in d_centroids.items()}

    def YOLO(self, img, conf, overlap_threshold, data_path , model_path):
        model = YOLO(model_path)
        _img = img.copy() # to make annotations
        # source image understands bgr cv2 format
        results = model.predict(task='detect', mode='predict', source=img, conf=conf, data=data_path, model=model_path)

        for r in results:
            
            num_predictions=r.boxes.shape[0]

            #conf, ball id and ball centroid will be stored in predictions
            predictions=np.zeros((num_predictions,8))
            annotator = Annotator(_img)
            
            boxes = r.boxes
            for id,box in enumerate(boxes):
                
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls #get the predicted class 
                conf = box.conf #get the confidence of thr prediction
                annotator.box_label(b, model.names[int(c)])
                
                x_top_left, y_top_left, x_bottom_right, y_bottom_right=b
                cx=(x_top_left+x_bottom_right)/2
                cy=(y_top_left+y_bottom_right)/2        
                #img = cv2.circle(img, (int(cx),int(cy)), radius=7, color=(0, 255, 0), thickness=-1)
                
                ball_number=int(model.names[int(c)])
                cx=float(cx)
                cy=float(cy)
                conf=float(conf)

                predictions[id,0]=conf
                predictions[id,1]=ball_number
                predictions[id,2]=cx
                predictions[id,3]=cy
                predictions[id,4]=x_bottom_right-x_top_left
                predictions[id,5]=y_bottom_right-y_top_left

        #img = annotator.result()  
        #cv2.imwrite('YOLO_Detection.jpg', img)     

        list_of_balls=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        valid_centroids=np.zeros((1,2))
        d_centroids={}
        d_annotations={}
        l_annotations=[]

        #we should take into account that predictions is already sorted from high conf to low
        for i,pred in enumerate(predictions):
            ball_number = pred[1]
            centroid = pred[2:4]
            conf=pred[0]
            h=pred[4]
            w=pred[5]
            
            #first prediction has a different treatment:
            if i==0:
                list_of_balls.remove(ball_number)
                valid_centroids[:]=centroid
                d_centroids[ball_number]=[centroid[0],centroid[1]]
                d_annotations['x']=int(centroid[0])
                d_annotations['y']=int(centroid[1])
                d_annotations['width']=int(w)
                d_annotations['height']=int(h)
                d_annotations['confidence']=conf
                d_annotations['class']=str(int(ball_number))
                l_annotations.append(d_annotations.copy())

            else:
                if ball_number in list_of_balls: #ball num not assigned yet
                    
                    distances=np.linalg.norm(centroid - valid_centroids, axis=1)
                    
                    if (distances>overlap_threshold).any():

                        list_of_balls.remove(ball_number)
                        valid_centroids = np.vstack([valid_centroids, centroid])
                        d_centroids[ball_number]=[centroid[0],centroid[1]]
                        d_annotations['x']=int(centroid[0])
                        d_annotations['y']=int(centroid[1])
                        d_annotations['width']=w
                        d_annotations['height']=h
                        d_annotations['confidence']=conf
                        d_annotations['class']=str(int(ball_number))
                        l_annotations.append(d_annotations.copy())

        return d_centroids, l_annotations

    def tune_ball_color(self,img,numbered_balls,color_space='hsv'):
        """
        """

        for i in np.unique(numbered_balls):
            if i!=0:
                masked_ball = np.zeros((img.shape[:-1]), dtype=np.uint8)#1 ch
                masked_ball[numbered_balls==i] = 255
                masked = cv2.bitwise_and(img, img, mask=masked_ball)
                contour_ball,_ = cv2.findContours(masked_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contour_ball[0])
                masked_ball=masked_ball[y:y+h, x:x+w]
                masked_ball_color = np.zeros_like(masked_ball)#1 ch
                masked=masked[y:y+h, x:x+w, :]
                palette=np.zeros((masked.shape), dtype=np.uint8)
                masked_hsv=cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
                masked_lab=cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)

                if color_space=='hsv':
                    lower = self.lower_hsv
                    upper = self.upper_hsv
                    thresholded_ball = cv2.inRange(masked_hsv, lower, upper)
                elif color_space=='lab':
                    lower = self.lower_lab
                    upper = self.upper_lab
                    thresholded_ball = cv2.inRange(masked_lab, lower, upper)
                    
                
                num_pixels_ball=np.count_nonzero((numbered_balls==i))
                num_white_pixels_ball=np.count_nonzero(thresholded_ball)
                proportion_white_pixels=num_white_pixels_ball/num_pixels_ball

                if 0.15<=proportion_white_pixels<0.85:
                    ball_type='striped'
                elif proportion_white_pixels<0.15:
                    ball_type='solid'
                else:
                    ball_type='cue ball'

                if ball_type != 'cue ball':
                    avg_lab=masked_lab[(thresholded_ball==0) & (masked_ball==255)].mean(axis=0)
                    colored_ball_pixels=masked[(thresholded_ball==0) & (masked_ball==255)]
                    avg=colored_ball_pixels.mean(axis=0)
                    palette[:]=avg.astype(np.uint8)
                    masked_ball_color[(thresholded_ball==0) & (masked_ball==255)]=255
                else:
                    avg_lab=masked_lab[(masked_ball==255)].mean(axis=0)
                    colored_ball_pixels=masked[(masked_ball==255)]
                    avg=colored_ball_pixels.mean(axis=0)
                    palette[:]=avg.astype(np.uint8)
                    masked_ball_color[(masked_ball==255)]=255

                measured_lab=np.tile(avg_lab, (9, 1))#there are 9 colors
                error=np.sqrt(np.sum(np.power(measured_lab[:,1:] - self.color_to_lab[:,1:], 2), axis=1)) #we use chroma instead of euclidean distance
                row_with_min_error=np.unravel_index(np.nanargmin(error, axis=None), error.shape)[0]
                ball_color=Eye.NUM_TO_COLOR[row_with_min_error]
                ball_number=Eye.COLOR_AND_TYPE_TO_NUM[ball_color][ball_type]

                
                fig = plt.figure()
                # show original image
                fig.add_subplot(151)
                fig.set_figwidth(10)
                fig.set_figheight(3)
                plt.title('lab histogram')
                color = ('k','r','yellow')
                for ch,col in enumerate(color):
                    histr = cv2.calcHist([masked_lab],[ch],None,[256],[1,256])
                    plt.plot(histr,color = col)
                    plt.xlim([0,256])
                    plt.title(f'ball {i+1}', fontsize=9)

                fig.add_subplot(152)
                plt.title(f'total ball pixels: {num_pixels_ball}', fontsize=9)
                plt.imshow(cv2.cvtColor(masked,cv2.COLOR_BGR2RGB))
                plt.axis('off')

                fig.add_subplot(153)
                plt.title(f'white pixels: {num_white_pixels_ball}', fontsize=9)
                plt.imshow(thresholded_ball,cmap=plt.cm.gray)
                plt.axis('off')
                
                fig.add_subplot(154)
                plt.title(f'colored pixels', fontsize=9)
                plt.imshow(masked_ball_color,cmap=plt.cm.gray)
                plt.axis('off')

                fig.add_subplot(155)
                plt.title(f'avg_lab: {int(avg_lab[0])}, {int(avg_lab[1])}, {int(avg_lab[2])}', fontsize=9)
                plt.imshow(cv2.cvtColor(palette,cv2.COLOR_BGR2RGB))
                plt.axis('off')

                fig.suptitle(f'ball: {ball_number}') 

                plt.show()


    def _find_ball_blobs_opencv(self,warp,connected_centroids):
        """
        Maybe faster than find_ball_blobs method
        """


        # noise removal
        #kernel = np.ones((3,3),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        opening = cv2.morphologyEx(connected_centroids,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(warp,markers)

        return markers
