import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

import skimage.io as io
import skimage.transform as trans


def find_division(div_raw):
  # blur = cv2.GaussianBlur(div_raw,(11,11),0)
  ret, thresh = cv2.threshold(div_raw, 137, 255, cv2.THRESH_BINARY)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
  dilation3 = cv2.dilate(thresh,kernel,iterations= 1)
  contours, _ = (cv2.findContours(dilation3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))[-2:]

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