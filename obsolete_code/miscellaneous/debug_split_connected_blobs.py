import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

thresh_3ch = cv2.imread('./results/blobs.png')
thresh = thresh_3ch[...,0]

# Compute Euclidean distance from every binary pixel
# to the nearest zero pixel then find peaks
distance_map = ndimage.distance_transform_edt(thresh)
local_max = peak_local_max(distance_map, min_distance=20, labels=thresh)
peaks_mask = np.zeros_like(distance_map, dtype=bool)
peaks_mask[tuple(local_max.T)] = True

# Perform connected component analysis then apply Watershed
markers = ndimage.label(peaks_mask, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=thresh)

# Iterate through unique labels
total_area = 0
for label in np.unique(labels):
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(thresh.shape, dtype="uint8")
    mask[labels == label] = 255

    # Find contours and determine contour area
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area += area
    cv2.drawContours(thresh_3ch, [c], -1, (36,255,12), 4)

print(total_area)
cv2.imwrite('./results/blobs_splitted.png', thresh_3ch)