import cv2
import json
from pool.ball_detection import Yolo

yolo=Yolo()
#read image with random ball config and background
for i in range(0,45):
    name=f'config_{i}'
    img = cv2.imread(f'./results/{name}.jpg')

    # Yolov8 results
    d_centroids, l_annotations = yolo.detect_balls(img,conf=0.25, overlap_threshold=100)
    d_annotation={}
    d_annotation['predictions']=l_annotations
    with open(f'./results/{name}.json', 'w') as fp:
        json.dump(d_annotation, fp)