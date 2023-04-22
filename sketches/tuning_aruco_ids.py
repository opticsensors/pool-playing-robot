import cv2

img = cv2.imread('./results/corners.jpg')

arucoDict=cv2.aruco.DICT_4X4_100
arucoDict = cv2.aruco.getPredefinedDictionary(arucoDict)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(
	arucoDict, arucoParams)

corners, ids, rejected = arucoDetector.detectMarkers(img)

if len(corners) > 0:
	# flatten the ArUco IDs list
	ids = ids.flatten()
	# loop over the detected ArUCo corners
	for (markerCorner, markerID) in zip(corners, ids):
		# extract the marker corners (which are always returned in
		# top-left, top-right, bottom-right, and bottom-left order)
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		# convert each of the (x, y)-coordinate pairs to integers
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))

		# draw the bounding box of the ArUCo detection
		cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
		cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
		cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)
		# compute and draw the center (x, y)-coordinates of the ArUco
		# marker
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
		# draw the ArUco marker ID on the img
		cv2.putText(img, str(markerID),
			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			2.5, (0, 255, 0), 2)
		print("[INFO] ArUco marker ID: {}".format(markerID))
		# show the output img
	cv2.imshow("img", cv2.resize(img,(0,0),fx=0.5,fy=0.5))
	cv2.waitKey(0)
cv2.imwrite('./results/aruco_ids.png', img)


