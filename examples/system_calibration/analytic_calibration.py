import numpy as np
import cv2
from pool.eye import Eye
from pool.calibration import DataExtractor

eye=Eye()
cameraMatrix=eye.cameraMatrix
dist=eye.dist
dist0=np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) # use this if we work with undistorted image

img = cv2.imread("./results/checkerboard.jpg")
img = eye.undistort_image(img, remapping=False)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

chessboardSize = (14,21)
frameSize = (img.shape[1],img.shape[0])

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

# If found, add object points, image points (after refining them)
if ret == True:
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    # Draw and display the corners
    cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
    cv2.imwrite('./results/checkerboard_detected_corners.jpg', img)

imgp=corners.reshape(-1,2) #same shape as objp

# Find 3D coordinates expressed in the world frame that are located
#  in a known plane (ball centroid plane) using 2D coordinates from image plane
found,rvec,tvec = cv2.solvePnP(objp, imgp, cameraMatrix, dist0)
rotMatrix = cv2.Rodrigues(rvec)[0]
rotMatrix_inv = np.linalg.inv(rotMatrix)
cameraMatrix_inv = np.linalg.inv(cameraMatrix) 

uvPoint=np.array([3950 ,  912, 1]).reshape(-1,1)
left_vector=rotMatrix_inv@cameraMatrix_inv@uvPoint
right_vector=rotMatrix_inv@tvec
z_const=0
s=(z_const+right_vector[2,0])/(left_vector[2,0])
xyzPoint=rotMatrix_inv@(s*cameraMatrix_inv@uvPoint-tvec)


# Find 2D coordinates in image plane given 3D coordinates 
# expressed in the world frame
xyz=rotMatrix@objp[0,:].reshape(-1,1) + tvec
x=xyz[0,0]/xyz[2,0]
y=xyz[1,0]/xyz[2,0]
fx=cameraMatrix[0,0]
fy=cameraMatrix[1,1]
cx=cameraMatrix[0,2]
cy=cameraMatrix[1,2]
u=fx*x+cx
v=fy*y+cy

# Find 2D coordinates in image plane given 3D coordinates 
# expressed in the world frame using opencv function
uvPoint_opencv=cv2.projectPoints(objp[0,:], rvec, tvec, cameraMatrix, dist0)[0]




# Find Carriage aruco point expressend in 3D world frame coordinates:
img_arucos=cv2.imread('./results/arucos_to_test_3.jpg')
img_arucos = eye.undistort_image(img_arucos, remapping=False)
img_pose=cv2.imread('./results/arucos_to_test_4.jpg')
img_pose = eye.undistort_image(img_pose, remapping=False)
img_corners=cv2.imread('./results/corners_0.jpg')
img_corners = eye.undistort_image(img_corners, remapping=False)
de = DataExtractor(img_corners)
flipper_arucos=[76, 77, 78, 79, 80, 81, 82, 83]
carriage_aruco=22
ball_centroid_arucos=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]
d_targets=eye.get_aruco_coordinates_given_several_aruco_ids(img_arucos,ball_centroid_arucos)
d_carriage_aruco=eye.get_aruco_coordinates_given_aruco_id(img_pose, carriage_aruco)

angle=de.get_flipper_angle(img_pose,flipper_arucos)
angle=angle-180

ball_centroid_uv=np.array(d_targets[14]).reshape(-1,1)
ball_centroid=np.ones((3,1))
ball_centroid[:2]=ball_centroid_uv
left_vector=rotMatrix_inv@cameraMatrix_inv@ball_centroid
right_vector=rotMatrix_inv@tvec
z_const=0
s=(z_const+right_vector[2,0])/(left_vector[2,0])
xyzPoint=rotMatrix_inv@(s*cameraMatrix_inv@ball_centroid-tvec)
trans=xyzPoint

angle=angle-90
Rt=np.array([[np.cos(angle), -np.sin(angle), 0, trans[0,0]],
            [np.sin(angle), np.cos(angle), 0, trans[1,0]],
            [0, 0, 1, trans[2,0]],
            [0, 0, 0, 1]])
a=26.4/25
b=74.1/25
h=220/25 - 19/25
d=22/25
offset_point_of_contact=np.array([-b,-a,-h,1]).reshape(-1,1)
offset_rotation_axis=np.array([d,0,0,1]).reshape(-1,1)
carriage_aruco_point3D = (Rt@offset_point_of_contact)[:3,0] + offset_rotation_axis[:3,0]
carriage_aruco_point2D=cv2.projectPoints(carriage_aruco_point3D, rvec, tvec, cameraMatrix, dist0)[0]

for a in np.arange(0.1,10,0.5):
    for b in np.arange(0.1,10,0.5):
        for d in np.arange(0.1,10,0.5):
            for h in np.arange(0.1,10,0.5):
                offset_point_of_contact=np.array([-b,-a,-h,1]).reshape(-1,1)
                offset_rotation_axis=np.array([d,0,0,1]).reshape(-1,1)
                carriage_aruco_point3D = (Rt@offset_point_of_contact)[:3,0] + offset_rotation_axis[:3,0]
                carriage_aruco_point2D=cv2.projectPoints(carriage_aruco_point3D, rvec, tvec, cameraMatrix, dist0)[0]
                if np.linalg.norm(carriage_aruco_point2D.ravel()-np.array([1065,1814]))<20:
                    print(a,b,d,h,carriage_aruco_point2D)
