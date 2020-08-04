import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

import skimage.io as io
import skimage.transform as trans

def mask_erosion(img):
  gray_set = set()
  w,h = img.shape[0], img.shape[1]
  for i in range(w):
    for j in range(h):
      if img[i][j] !=0:
        gray_set.add(img[i][j])
  gray_set = list(gray_set)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27))
  out = np.zeros((w,h),int)
  gray_cells = []

  for gray_value in gray_set:
    temp_img = np.zeros((w,h),int)
    for i in range(w):
      for j in range(h):
        if img[i][j] == gray_value:
          temp_img[i][j] = gray_value

    erosion = cv2.erode(temp_img.astype(np.uint8),kernel,iterations= 1)

    for m in range(w):
      for n in range(h):
        if erosion[m][n] !=0:
          out[m][n] = erosion[m][n]
  return out

'''
img = cv2.imread('/content/drive/Colab/data/DIC-C2DH-HeLa-train/01_ST/SEG/man_seg000.tif',-1)
img_erosion = mask_erosion(img)
cv2.imwrite('/content/drive/Colab/t000_marker.tif',img_erosion)

plt.figure(figsize=(15,8))
plt.subplot(131)
plt.imshow(img,'gray')
plt.subplot(132)
plt.imshow(img_erosion,'gray')
plt.show()
'''