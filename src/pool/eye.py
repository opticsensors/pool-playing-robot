import cv2
import numpy as np
import diplib as dip
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

class Eye(object):
    """

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
        'white': {'cue ball': 0},
        'yellow': {'solid':1, 'striped':9},
        'blue': {'solid':2, 'striped':10},
        'red': {'solid':3, 'striped':11},
        'purple': {'solid':4, 'striped':12},
        'orange': {'solid':5, 'striped':13},
        'green': {'solid':6, 'striped':14},
        'burgundy': {'solid':7, 'striped':15},
        'black': {'solid':8}
    }
    
    #lab calibration results of ball colors
    #sorted by rows like: white, yellow, blue, red,... (same order as pool balls)
    COLOR_TO_LAB={
        'white': [227, 130, 148],
        'yellow': [216.18407452, 124.80355131, 195.73519079],
        'blue': [89.64986175, 134.93342045, 103.84520826],
        'red': [147.45829885, 172.49603901, 178.22393935],
        'purple': [78.6311673, 135.5586135, 123.057679],
        'orange': [180.62554741, 151.8795586 , 187.98600759],
        'green': [110.05696026, 106.50088757, 132.41375192],
        'burgundy': [121.56706638, 160.19773476, 158.41379201],
        'black': [58, 129, 136]
    }
    
    WHITE_LOWER_LAB=[175, 0, 0]

    WHITE_UPPER_LAB=[255, 147, 164]

    WHITE_LOWER_HSV=[0, 0, 179]

    WHITE_UPPER_HSV=[180, 106, 255]
    
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

    def color_segmentation(self,hsv, lower_color, upper_color,filter_radius):
        # Threshold the HSV image to get only cloth colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        #use a median filter to get rid of speckle noise
        median = cv2.medianBlur(mask,filter_radius)
        
        inverted_mask=cv2.bitwise_not(mask)
        inverted_median = cv2.bitwise_not(median)

        return inverted_mask,inverted_median

    def perspective_transform(self,image, corners):

        # Order points in clockwise order:

        # First, we separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        #corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        ordered_corners= (top_l, top_r, bottom_r, bottom_l)

        top_l, top_r, bottom_r, bottom_l = ordered_corners

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
        ordered_corners = np.array(ordered_corners, dtype="float32")

        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        # Return the transformed image
        return cv2.warpPerspective(image, matrix, (width, height))


    def find_ball_bolbs(self,thresh,connectivity,min_size, max_size, thresh_convexity, thresh_roundness):
        """
        Finds all image blobs that are below an area threshold and gets rid of them.
        Those dots aren't considered part of the pattern and they are classified as noise.
        
        Parameters
        ----------
            thresh: npy array
                Thresholded image (this image is binary, i.e., it only contains pixels
                values of 0 or 255).
            dot_min_size: int
                Minimum area that a blob can have and not be classified as noise.
        
        """
        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, 
                                                                                    connectivity=connectivity)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        #sizes = stats[1:, -1] 
        #centroids = centroids[1:, :]
        #nb_components = nb_components - 1
        
        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        
        sizes = stats[:, -1] 
        numbered_single_blobs=np.zeros((output.shape), dtype=np.uint8)
        #numbered_connected_blobs=np.zeros((output.shape), dtype=np.uint8)
        connected_blobs=np.zeros((output.shape), dtype=np.uint8)
        single_blobs=np.zeros((output.shape), dtype=np.uint8)

        d_single_centroids={}
        single_ball_number=1
        #d_connected_centroids={}
        #connected_ball_number=1

        #for every component in the image, you keep it only if it's between min_size and max_size
        for i in range(1, nb_components): #we satrt at 1 because first element is bg

            blob= np.zeros((thresh.shape), dtype=np.uint8)#1 ch
            blob[output==i] = 255
            table = dip.MeasurementTool.Measure(blob, features=["Convexity", "Roundness"])
            table=np.array(table)
            convexity=table[:,0]
            roundness=table[:,1]

            if min_size<sizes[i] <max_size:

                if convexity<thresh_convexity :
                    connected_blobs[output == i]=255
                    #numbered_connected_blobs, d_connected_centroids= self.split_connected_balls_v2(connected_blobs)
                    #d_connected_centroids[connected_ball_number]=list(centroids[i])
                    #connected_ball_number+=1
                if roundness>thresh_roundness:
                    numbered_single_blobs[output == i] = single_ball_number
                    single_blobs[output == i] = 255
                    d_single_centroids[single_ball_number]=list(centroids[i])
                    single_ball_number+=1
                #print(i,convexity, roundness)

        return numbered_single_blobs, d_single_centroids, single_blobs,connected_blobs

    def classify_balls(self,img,numbered_balls,d_centroids,color_space='hsv'):

        #new_numbered_balls=np.zeros_like(numbered_balls)
        old_to_new={}
        #number for detected balls
        unique,_=np.unique(numbered_balls, return_counts=True)
        l_unique=[ball_num for ball_num in unique if ball_num!=0]

        for i in l_unique:

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

            if 0.15<=proportion_white_pixels<0.85:
                ball_type='striped'
            elif proportion_white_pixels<0.15:
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

        return {old_to_new[k]: v for k, v in d_centroids.items()}

    def split_connected_balls(self,warp,connected_centroids):

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

    def split_connected_balls_v2(self,thresh):

        d_connected_centroids={}

        # Compute Euclidean distance from every binary pixel
        # to the nearest zero pixel then find peaks
        distance_map = ndimage.distance_transform_edt(thresh)
        local_max = peak_local_max(distance_map, indices=False, min_distance=20, labels=thresh)

        # Perform connected component analysis then apply Watershed
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance_map, markers, mask=thresh)
        l_unique=[ball_num for ball_num in labels if ball_num!=0]

        for i in l_unique:
            masked_ball = np.zeros((thresh.shape), dtype=np.uint8)#1 ch
            masked_ball[labels==i] = 255
            M = cv2.moments(masked_ball)
            # calculate x,y coordinate of center
            centroid= [(M["m10"] / M["m00"]),(M["m01"] / M["m00"])]
            d_connected_centroids[i]=centroid

        return labels,d_connected_centroids
