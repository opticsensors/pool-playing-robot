import cv2
import json
from pool.eye import Eye

#read image with random ball config and background
for i in range(0,45):
    name=f'config_{i}'
    img = cv2.imread(f'./results/{name}.jpg')

    eye=Eye()
    eye.bottom_aruco_ids=[17,8]
    eye.top_aruco_ids=[9,3]
    eye.left_aruco_ids=[1,11]
    eye.right_aruco_ids=[23,10]

    # if images are not transformed:
    #if i == 0:
    #    corners=eye.get_pool_corners(img)

    #warp=eye.perspective_transform(img,corners)
    #cv2.imwrite(f'./results/warp_{name}.jpg', warp)

    # Yolov8 results
    d_centroids, l_annotations=eye.YOLO(img,conf=0.25, overlap_threshold=100, data_path='./data/data.yaml',model_path='./data/yolov8m.pt')
    
    # if we want to see the results:
    #for ball_num in d_centroids:
    #    x,y=d_centroids[ball_num]
    #    img=cv2.putText(img, "#{}".format(ball_num), (int(x) - 10, int(y)),
    #    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    #cv2.imwrite('./results/chosen_YOLO_Detection.jpg', img)

    d_annotation={}
    d_annotation['predictions']=l_annotations
    with open(f'./results/{name}.json', 'w') as fp:
        json.dump(d_annotation, fp)