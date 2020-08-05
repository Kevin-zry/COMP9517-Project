import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

import skimage.io as io
import skimage.transform as trans

def draw_contours(img,pre):
  pre_float = np.zeros((img.shape[0],img.shape[1]),float)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      pre_float[i][j] = pre[i][j] *255
  pre_uint = pre_float.astype(np.uint8)


  ret, thresh = cv2.threshold(pre_uint, 127, 255, cv2.THRESH_BINARY)
  contours_raw, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

  ellipse_list = []
  for i in range(len(contours)):
    x,y = int(bound[i][0]), int(bound[i][1])
    w,h = int(bound[i][2]), int(bound[i][3])
    cv2.rectangle(rec, (x,y), (x+w,y+h), (0, 0, 255), 2)
    ellipse = cv2.fitEllipse(contours[i])
    ellipse_list.append(ellipse)
    cv2.ellipse(rec, ((x+w//2,y+h//2), (min(h,w),max(h,w)), ellipse[2]), (0, 0, 255), 2)
    

  tex = rec.copy()
  for i,j in zip(contours,range(len(contours))):
      M = cv2.moments(i)
      x,y = int(center[j][0]), int(center[j][1])
      w,h = int(bound[j][2]), int(bound[j][3])
      #cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) → None
      draw = cv2.putText(tex, str(j), (x,y), 1, 3, (255, 0, 255), 3)

  # draw_with_cellnum = cv2.putText(draw, "CellNum: "+str(len(center)), (int(img.shape[0])//50,int(img.shape[1]//17)), 1, 2, (255, 0, 255), 3)
  return draw, bound


######################################################
save_path = ""
raw_data = np.load(save_path + "s1_raw.npy")
pre_data = np.load(save_path + "s1_pre.npy")


res = {}
for i in range(10):
  img = raw_data[i]
  pre = pre_data[i]

  draw, boxs = draw_contours(img,pre)
  # cv2.imwrite(save_path + '/t' + num[1:] + '_predict.tif',draw)
  res[i] = {}
  res[i]['frame'] = i
  res[i]['draw_img'] = draw
  res[i]['cell_num'] = len(boxs)
  res[i]['boxs'] = boxs
  
  box_centers = []
  for j in range(len(boxs)):
    x,y = int(boxs[j][0]), int(boxs[j][1])
    w,h = int(boxs[j][2]), int(boxs[j][3])
    box_centers.append((x+w//2,y+h//2))
  
  res[i]['centers'] = box_centers

  print("frame:",i)
  print(box_centers)
  
  plt.figure(figsize=(15,8))
  plt.subplot(141)
  plt.imshow(img,'gray')
  plt.subplot(142)
  plt.imshow(pre,'gray')
  plt.subplot(143)
  plt.imshow(draw,'gray')
  # plt.savefig('/content/drive/Colab/predict/try3_s1_plot/t'+ num[1:] + '.jpg')
  plt.show()