import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

import skimage.io as io
import skimage.transform as trans



def not_close_to_boundary(point, th=0):
    low, high = th, 512 - th
    # print('point:', point)
    if point[0] < low or point[0] > high:
        return False
    if point[1] < low or point[1] > high:
        return False
    return True

def draw_contours(img,pre,frame):
  pre_float = np.zeros((img.shape[0],img.shape[1]),float)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      pre_float[i][j] = pre[i][j] *255
  pre_uint = pre_float.astype(np.uint8)


  ret, thresh = cv2.threshold(pre_uint, 127, 255, cv2.THRESH_BINARY)

  cv2.imwrite("/content/drive/Colab/example/res.jpg",thresh)
  
  contours_raw, _ = (cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))[-2:]

  contours_final = []
  for i in contours_raw:
    if (cv2.contourArea(i)>1000):
      contours_final.append(i)

  out = img.copy()
  contours = contours_final

  contours_appproxPoly = [None]*len(contours)
  bound = [None]*len(contours)
  center = [None]*len(contours)
  radius = [None]*len(contours)

  for i, cnt in enumerate(contours):
    # cv2.approxPolyDP(curve, epsilon, closed[, approxCurve]) → approxCurve
    contours_appproxPoly[i] = cv2.approxPolyDP(cnt, 1, True)
    # cv.BoundingRect(points, update=0) → CvRect
    bound[i] = cv2.boundingRect(contours_appproxPoly[i])
    # cv2.minEnclosingCircle(points) → center, radius
    center[i], radius[i] = cv2.minEnclosingCircle(contours_appproxPoly[i])

  rec = out.copy()

  rect_list = []
  rect_out_list = []
  for i in range(len(contours)):
    x,y = int(bound[i][0]), int(bound[i][1])
    w,h = int(bound[i][2]), int(bound[i][3])
    #cv2.rectangle(rec, (x,y), (x+w,y+h), (0, 0, 255), 2)
    #ellipse = cv2.fitEllipse(contours[i])
    #cv2.ellipse(rec, ((x+w//2,y+h//2), (min(h,w),max(h,w)), ellipse[2]), (0, 0, 255), 2)
    rect = cv2.minAreaRect(contours[i])
    
    if not_close_to_boundary(rect[0]):
      rect_list.append(rect)
      rect_out = [frame, len(rect_out_list), (round(rect[0][0]),round(rect[0][1])), rect[1], rect[2]]
      rect_out_list.append(rect_out)
      cv2.rectangle(rec, (int(rect[0][0]-w//2), int(rect[0][1]-h//2)), (x+w,y+h), (0, 0, 255), 2)
      #b = cv2.boxPoints(rect)
      #b = np.int0(b)
      #cv2.drawContours(rec,[b],0,(0,0,255),2)
    

  tex = rec.copy()
  for i,j in zip(contours,range(len(rect_list))):
      #M = cv2.moments(i)
      #x,y = int(center[j][0]), int(center[j][1])
      #w,h = int(bound[j][2]), int(bound[j][3])
      x,y = int(rect_list[j][0][0]),int(rect_list[j][0][1])


  draw_with_cellnum = cv2.putText(rec, "CellNum: "+str(len(center)), (int(img.shape[0])//50,int(img.shape[1]//17)), 1, 2, (255, 0, 255), 3)
  return rec, bound, rect_list, rect_out_list