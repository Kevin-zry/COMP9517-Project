import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

import skimage.io as io
import skimage.transform as trans

from math import sqrt

save_path = ""


def find_division(div_raw):
  # blur = cv2.GaussianBlur(div_raw,(11,11),0)
  ret, thresh = cv2.threshold(div_raw, 137, 255, cv2.THRESH_BINARY)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
  dilation3 = cv2.dilate(thresh,kernel,iterations= 1)
  contours, _ = cv2.findContours(dilation3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  out_raw = div_raw.copy()
  cv2.drawContours(out_raw,contours,-1,(0,0,255),5)

  c_area = []
  contours_final = [] 
  for i in contours:
    c_area.append(cv2.contourArea(i))
  c_area = sorted(c_area, reverse = True)
  #print(c_area[:5])

  for i in contours:
    #if cv2.contourArea(i)==max(c_area):
    if 6000<cv2.contourArea(i)<10000:
      contours_final.append(i)

  out = div_raw.copy()
  if contours_final:
    cv2.drawContours(out,contours_final,-1,(0,0,255),10)

  return out,contours_final


def find_status(circles,point):
  for c in circles:
    if cv2.pointPolygonTest(c, point, False) == 1:
      return 1
  else:
    return 0

######################################################################

raw_data = np.load(save_path + "s1_raw.npy")
pre_data = np.load(save_path + "s1_pre.npy")
res_data = np.load(save_path + 's1_draw_res.npy',allow_pickle=True)
print(len(res_data))
print(res_data[0])


for i in range(84):
  print("frame:",i)
  img = raw_data[i]
  out,black_circles = find_division(raw_data[i])
  print("black circle num",len(black_circles))

  fit_list = res_data[i]['fitRect']
  centers = []
  
  #for rect in fit_list:
    #centers.append(rect[1])
  # [27, (151.5, 476.5), (99.0, 69.0), 0.0]
  # print(centers)
  
  division = []
  for rect in fit_list:
    if find_status(black_circles, rect[1]): # 中心在黑圈里
      division.append(rect)
  print("division info:",division)

  draw = res_data[i]['draw_img'].copy()
  if division:
    for div in division:
      #print(div[1:])
      b = cv2.boxPoints((div[1],div[2],0))
      b = np.int0(b)
      cv2.drawContours(draw,[b],0,(0,0,255),15)

  plt.figure(figsize=(15,8))
  plt.subplot(151)
  plt.imshow(img,'gray')
  plt.subplot(152)
  plt.imshow(out,'gray')
  plt.subplot(153)
  plt.imshow(draw,'gray')
  plt.show()