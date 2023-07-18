import cv2
from pool.eye import Eye

eye=Eye()
img = cv2.imread('./results/end_effector.jpg')
aruco_coord=eye.find_all_aruco_coordinates(img, debug_path='./results/', debug=True)
print(aruco_coord)
