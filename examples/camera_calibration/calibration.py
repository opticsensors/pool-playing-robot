import numpy as np
import cv2 as cv
import os
from pool.utils import Params

data_folder='data3'

if data_folder == 'data1':
    valid_images=[0,1,2,3,5,6,9,10,11,12,13,14,15,16,17,18,19,21,23,24,25]
    chessboardSize = (6,9)
elif data_folder == 'data2':
    chessboardSize = (14,21)
    valid_images=[1,2,3,4,5,6,7,8,9,11,13,16,17,19,20,21,23,24,27,28,30,34,35,36,37,39,40,42,43,46,47,48,49,50,51,52,53,54,56,57,60,61,62]
elif data_folder == 'data3':
    valid_images=[0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78]
    chessboardSize = (6,9)

f = open(f"./results/{data_folder}/calibration_results.txt", 'w')

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

frameSize = (5184,3456)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# bad images are not considered (manually removed)
for count in valid_images:
    img = cv.imread(f"./results/{data_folder}/img_{count}.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    print(count, ret)
    
    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', cv.resize(img, (0,0), fx=0.15, fy=0.15))
        cv.waitKey(300)

############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print('Camera calibrated:', ret, file=f)
print("\nCamera Matrix:\n", cameraMatrix, file=f)
print("\nDistortion params:\n", dist, file=f)

path_to_repo=Params().PATH_REPO
np.save(os.path.join(path_to_repo,'data','cameraMatrix.npy'), cameraMatrix)
np.save(os.path.join(path_to_repo,'data','dist.npy'), dist)

# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "\nReprojection Error:\n", mean_error/len(objpoints), file=f)

f.close()