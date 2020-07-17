import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

def contrast_stretch(img):
    MIN, MAX = np.amin(img), np.amax(img)
    return np.uint8((img - MIN) * 255/(MAX - MIN))


p0 = cv2.imread('PhC-C2DL-PSC/t000.tif', 0)
#plt.imshow(p0,'gray')

# Step 1 - Binarize via thresholding
ret, th1 = cv2.threshold(p0,185,255,cv2.THRESH_BINARY)
# plt.imshow(th1,'gray')

# Step 2 - Calculate the distance transform
distance = ndi.distance_transform_edt(th1)

# Step 3 - Generate the watershed markers
local_max = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=th1)
markers = ndi.label(local_max)[0]

# Step 4 - Perform watershed and store the labels
ws_labels = watershed(-distance, markers, mask=th1)

# plt.imshow(-distance)
# plt.imshow(ws_labels)


seg = contrast_stretch(ws_labels)

contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cell_ctrs = [c for c in contours if cv2.contourArea(c) > 25]
draw = cv2.drawContours(p0, cell_ctrs,-1,(0,255,0),1)
# cv2.imshow('contours', draw)
# cv2.waitKey(0)

print(len(cell_ctrs))

for i,j in zip(cell_ctrs,range(len(cell_ctrs))):
    M = cv2.moments(i)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    draw1=cv2.putText(draw, str(j), (cX, cY), 1,1, (255, 0, 255), 1)
# plt.imshow(draw1,'gray')
cv2.imshow('labeled contours', draw1)
cv2.waitKey(0)