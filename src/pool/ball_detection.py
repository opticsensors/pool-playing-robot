import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
from pool.utils import Params

class ClassicCV:
    """
    A class that detects the centroids and numbers of pool balls given an image using classic computer vision algorithms
    """
    def __init__(self):
        self.params = Params()
        self.lower_hsv = np.array(self.params.WHITE_LOWER_HSV)
        self.upper_hsv = np.array(self.params.WHITE_UPPER_HSV)
        self.lower_lab = np.array(self.params.WHITE_LOWER_LAB)
        self.upper_lab = np.array(self.params.WHITE_UPPER_LAB)
        self.cloth_lower_hsv = np.array(self.params.CLOTH_LOWER_HSV)
        self.cloth_upper_hsv = np.array(self.params.CLOTH_UPPER_HSV)
        self.color_to_lab = self._dict_to_arr(self.params.COLOR_TO_LAB)

    def _dict_to_arr(self,color_to_lab):
        color_to_num = {v: k for k, v in self.params.NUM_TO_COLOR.items()}
        num_to_lab={color_to_num[k]: v for k, v in color_to_lab.items()}
        lab_arr=np.zeros((len(color_to_lab),3))
        for num in num_to_lab:
            lab=num_to_lab[num]
            lab_arr[num,:]=np.array(lab)
        
        return lab_arr

    def get_cloth_color(self,hsv,search_width=45):
        """
        Find the most common HSV values in the image.
        In a well lit image, this will be the cloth
        """

        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]) 
        h_max = np.unravel_index(np.nanargmax(hist, axis=None), hist.shape)[0]
        
        hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        s_max = np.unravel_index(np.nanargmax(hist, axis=None), hist.shape)[0]
        
        hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        v_max = np.unravel_index(np.nanargmax(hist, axis=None), hist.shape)[0]

        # define range of cloth color in HSV 
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

    def crop_image(self, img, h_offset, v_offset):
        """
        Crop image given offsets (distance in pixels between old and new image borders)
        """
        return img[v_offset:-v_offset, h_offset:-h_offset]

    def substract_background(self, thresh, bg_mask):
        """
        Removes the white blob of the background in the binary image with the balls
        """
        thresh[bg_mask==255]=0 # TODO crop bg image or image with balls so they have same dimensions 
        return thresh

    def remove_small_dots(self,thresh,connectivity):
        """
        Finds all image blobs that are below or between an area threshold and gets rid of them.
        Those dots aren't considered part of the pool balls and they are classified as noise.
        
        """
        #define min size
        factor_of_safety=0.15
        H=thresh.shape[1]
        V=thresh.shape[0]
        min_size=factor_of_safety*H*V*self.params.RATIO_BALL_RECTANGLE 

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=connectivity)        
        #size of all found blobs (the first one is the background)
        sizes = stats[:, -1] 
        # we will save blobs bigger than min size here
        blobs=np.zeros((output.shape), dtype=np.uint8)

        #we satrt at 1 because first element is bg
        for i in range(1, nb_components): 
            # check if blob is bigger than min size
            if sizes[i]>min_size:
                blobs[output==i] = 255
        return blobs

    def find_ball_blobs(self,thresh):
        """
        Given a thresholded image with ball blobs, this method splits 
        connected blobs using watershed algorithm
        """
        d_centroids={}
        # Compute Euclidean distance from every binary pixel
        # to the nearest zero pixel then find peaks
        distance_map = ndimage.distance_transform_edt(thresh)
        local_max = peak_local_max(distance_map, min_distance=20, labels=thresh)
        peaks_mask = np.zeros_like(distance_map, dtype=bool)
        peaks_mask[tuple(local_max.T)] = True

        # Perform connected component analysis then apply Watershed
        markers = ndimage.label(peaks_mask, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance_map, markers, mask=thresh)

        for i in np.unique(labels):
            if i!=0:
                masked_ball = np.zeros((thresh.shape), dtype=np.uint8)#1 ch
                masked_ball[labels==i] = 255
                M = cv2.moments(masked_ball)
                # calculate x,y coordinate of center
                centroid= [(M["m10"] / M["m00"]),(M["m01"] / M["m00"])]
                d_centroids[i]=centroid

        return labels,d_centroids
    
    def debug_find_ball_blobs(self, blobs, numbered_blobs):
        """
        Draws contour of the splitted ball blobs
        """
        blobs_3ch=cv2.merge((blobs,blobs,blobs))
        for label in np.unique(numbered_blobs):
            if label == 0:
                continue

            # Create a mask
            mask = np.zeros(blobs.shape, dtype="uint8")
            mask[numbered_blobs == label] = 255

            # Find contours and determine contour area
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(blobs_3ch, [c], -1, (0,0,255), 12)
        return blobs_3ch

    def classify_balls(self,img,numbered_balls,d_centroids,color_space='hsv'):
        """
        For each labeled blob, this method decides its color (white, yellow, ...) and its type (solid, striped)
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

                if self.params.LOWER_PROPORTION_WHITE_PIXELS<=proportion_white_pixels<self.params.UPPER_PROPORTION_WHITE_PIXELS:
                    ball_type='striped'
                elif proportion_white_pixels<self.params.LOWER_PROPORTION_WHITE_PIXELS:
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
                ball_color=self.params.NUM_TO_COLOR[row_with_min_error]
                try:
                    ball_number=self.params.COLOR_AND_TYPE_TO_NUM[ball_color][ball_type]
                except KeyError:
                    ball_number=-1
                old_to_new[i]=ball_number

        return {old_to_new[k]: v for k, v in d_centroids.items()}

    def display_images_of_same_size_in_a_grid(self, l_images, grid_size, titles, margin=50, spacing =35, dpi=100.):

        rows, cols= grid_size
        max_h, max_w = l_images[0].shape[:2]
        width = (cols*max_w+cols*margin+spacing)/dpi # inches
        height= (rows*max_h+rows*margin+spacing)/dpi

        left = margin/dpi/width #axes ratio
        bottom = margin/dpi/height
        wspace = spacing/float(max_w)

        fig, axes  = plt.subplots(rows,cols, figsize=(width,height), dpi=dpi)
        fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                            wspace=wspace, hspace=wspace)

        for ax, im, name in zip(axes.flatten(),l_images, titles):
            ax.axis('off')
            ax.set_title(name)
            ax.imshow(im)
        # save figure to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close('all')
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    def tune_white_color(self,img,numbered_balls):
        """
        Generates a grid of images of the balls, that later can be used to
        manually tune the hsv or lab color ranges to mask white pixels 
        """
        max_h=0
        max_w=0
        l_images=[]
        for i in np.unique(numbered_balls):
            if i!=0:
                masked_ball = np.zeros((img.shape[:-1]), dtype=np.uint8)#1 ch
                masked_ball[numbered_balls==i] = 255
                masked = cv2.bitwise_and(img, img, mask=masked_ball)
                contour_ball,_ = cv2.findContours(masked_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contour_ball[0])
                masked=masked[y:y+h, x:x+w, :]
                masked=cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
                h,w=masked.shape[:2]
                if max_h<h:
                    max_h=h
                if max_w<w:
                    max_w=w
                l_images.append(masked)
        # make image the same size
        titles=[]
        for i,img in enumerate(l_images):
            size=img.shape[:2]
            delta_w = max_w - size[1]
            delta_h = max_h - size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            color = [0, 0, 0]
            new_img = cv2.copyMakeBorder(img, top, bottom, 
                                         left, right, 
                                         cv2.BORDER_CONSTANT,
                                         value=color)
            l_images[i] = new_img
            titles.append(f'ball_id: {i+1}')

        return self.display_images_of_same_size_in_a_grid(l_images, (4,4), titles)
    
    def tune_ball_color(self,img,numbered_balls,color_space='hsv'):
        """
        Generates a grid of images of the balls, their color and white pixel masks, 
        and the avg color of the balls for debugging purposes and color tuning
        """
        l_images=[]
        titles=[]
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

                if self.params.LOWER_PROPORTION_WHITE_PIXELS<=proportion_white_pixels<self.params.UPPER_PROPORTION_WHITE_PIXELS:
                    ball_type='striped'
                elif proportion_white_pixels<self.params.LOWER_PROPORTION_WHITE_PIXELS:
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
                ball_color=self.params.NUM_TO_COLOR[row_with_min_error]
                try:
                    ball_number=self.params.COLOR_AND_TYPE_TO_NUM[ball_color][ball_type]
                except KeyError:
                    ball_number=-1
                thresholded_ball_3ch=cv2.merge((thresholded_ball,thresholded_ball,thresholded_ball))
                masked_ball_color_3ch=cv2.merge((masked_ball_color,masked_ball_color,masked_ball_color))
                l_sub_images=[masked,thresholded_ball_3ch,masked_ball_color_3ch,palette]
                sub_titles=[f'total ball pixels: {num_pixels_ball}', f'white pixels: {num_white_pixels_ball}', f'colored pixels', f'avg_lab: {int(avg_lab[0])}, {int(avg_lab[1])}, {int(avg_lab[2])}']
                l_images.append(self.display_images_of_same_size_in_a_grid(l_sub_images, (1,4), sub_titles))
                titles.append(f'ball_id: {i} | ball_prediction: {ball_number}')
                print(ball_number,avg.astype(np.uint8))
        return self.display_images_of_same_size_in_a_grid(l_images, (4,4), titles)
    
    def detect_balls(self, warp, warp_bg, cloth_color, color_space, debug_path='./results/', debug=False):
        """
        Wrapper method tha uses the above methods to compute the the centroids and classify the balls
        A debug option is available to see what is going on step by step 
        """
        hsv=cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        bg_hsv=cv2.cvtColor(warp_bg, cv2.COLOR_BGR2HSV)
        
        if cloth_color == 'auto':
            lower_color, upper_color = self.get_cloth_color(hsv,search_width=30)
        elif cloth_color == 'tuned':
            lower_color=self.cloth_lower_hsv
            upper_color=self.cloth_upper_hsv

        mask,mask_processed =self.color_segmentation(hsv, lower_color, upper_color)
        bg_mask,bg_mask_processed =self.color_segmentation(bg_hsv, lower_color, upper_color)
        mask_without_bg=self.substract_background(mask_processed,bg_mask_processed)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
        mask_without_bg_processed = cv2.morphologyEx(mask_without_bg, cv2.MORPH_OPEN, kernel,iterations = 2)
        blobs = self.remove_small_dots(mask_without_bg_processed,connectivity=8) 
        numbered_blobs,d_centroids = self.find_ball_blobs(blobs)

        if cv2.countNonZero(blobs)!=0:
            sorted_centroids=self.classify_balls(warp,numbered_blobs,d_centroids,color_space=color_space)

        if debug:
            #tune white color
            results_tune_white=self.tune_white_color(warp, numbered_blobs)
            # tune ball color
            results_tune_color=self.tune_ball_color(warp,numbered_blobs,color_space=color_space)
            blobs_with_contours_drawn = self.debug_find_ball_blobs(blobs, numbered_blobs)

            cv2.imwrite(f'{debug_path}mask.png', mask)
            cv2.imwrite(f'{debug_path}mask_processed.png', mask_processed)
            cv2.imwrite(f'{debug_path}bg_mask.png', bg_mask)
            cv2.imwrite(f'{debug_path}bg_mask_processed.png', bg_mask_processed)
            cv2.imwrite(f'{debug_path}mask_without_bg.png', mask_without_bg)
            cv2.imwrite(f'{debug_path}mask_without_bg_processed.png', mask_without_bg_processed)
            cv2.imwrite(f'{debug_path}blobs.png', blobs)
            cv2.imwrite(f'{debug_path}blobs_with_contours_drawn.png', blobs_with_contours_drawn)
            cv2.imwrite(f'{debug_path}results_tune_white.png', results_tune_white)
            cv2.imwrite(f'{debug_path}results_tune_color.png', results_tune_color)
        
        return sorted_centroids

    def debug(self,img, d_centroids):
        """
        Draws the centroids and number of the detected balls 
        """
        img_to_draw=img.copy()
        for ball_num in d_centroids:
            x,y=d_centroids[ball_num]
            cv2.putText(img_to_draw, "#{}".format(ball_num), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.circle(img_to_draw, (int(x), int(y)), 8, (255, 0, 255), -1)
        return img_to_draw
    
class Yolo:
    """
    A class that detects the centroids and numbers of pool balls given an image using yolo_v8 medium algorithm
    A custom annotated dataset is used to train the weights
    """
    def __init__(self, data_path=None, model_path=None):
        params=Params()
        if data_path is None:
            self.data_path = os.path.join(params.PATH_REPO, 'data', 'data.yaml')
        else:
            self.data_path = data_path
        if model_path is None:
            self.model_path = os.path.join(params.PATH_REPO, 'data', 'yolov8m_version9.pt')
        else:
            self.model_path = model_path

    def detect_balls(self, img, conf, overlap_threshold):
        """
        Uses Yolo algorithm to compute the the centroids and classify the balls
        The impossibility of two balls from the same class exsisting in the image is added
        """
        model = YOLO(self.model_path)
        _img = img.copy() # to make annotations
        # source image understands bgr cv2 format
        results = model.predict(task='detect', 
                                mode='predict', 
                                source=img, 
                                conf=conf, 
                                data=self.data_path, 
                                model=self.model_path)

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
    
    def debug(self, img, d_centroids):
        """
        draws the centroids and number of the detected balls 
        """
        img_to_draw=img.copy()
        for ball_num in d_centroids:
            x,y=d_centroids[ball_num]
            cv2.putText(img_to_draw, "#{}".format(int(ball_num)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.circle(img_to_draw, (int(x), int(y)), 8, (255, 0, 255), -1)
        return img_to_draw