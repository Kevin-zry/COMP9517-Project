import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

import skimage.io as io
import skimage.transform as trans

import os

#-------------- Please change your mask path here -------------#
mask_path = '/content/drive/Colab/data/DIC-C2DH-HeLa/'
# ------------------------------------------------------------ #
 

def mkdir(path):
 
  folder = os.path.exists(path)
  if not folder:       
    os.makedirs(path)       
    
mkdir(mask_path + '01_new_mask/')  
mkdir(mask_path + '02_new_mask/')  


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


# get all eroded masks of the sequence 1
mask_path1 = mask_path + '01_ST/SEG/' # Please also set the mask path
mask_data1 = np.asarray(list(io.ImageCollection(mask_path1 + 'man_seg*.tif')))


mask_new_path1 = mask_path + '01_new_mask/'
#for i in range(mask_data.shape[0]):
print("erode mask 1...")
print(mask_data1.shape)

for i in range(mask_data1.shape[0]):
#for i in range(1):
  num = str(1000+i)
  img = mask_data1[i]
  img_erosion = mask_erosion(img)
  cv2.imwrite(mask_new_path1 + 't' + num[1:] +'_marker.tif', img_erosion)
print("mask 1 done!")
print()



print("erode mask 2...")
mask_path2 = mask_path + '02_ST/SEG/' # Please also set the mask path
mask_data2 = np.asarray(list(io.ImageCollection(mask_path2 + 'man_seg*.tif')))
print(mask_data2.shape)

mask_new_path2 = mask_path + '02_new_mask/'
for i in range(mask_data2.shape[0]):
#for i in range(1):
  num = str(1000+i)
  img = mask_data2[i]
  img_erosion = mask_erosion(img)
  cv2.imwrite(mask_new_path2 + 't' + num[1:] +'_marker.tif', img_erosion)


print("mask 2 done!")
print()
print("all done, Thank you!")
