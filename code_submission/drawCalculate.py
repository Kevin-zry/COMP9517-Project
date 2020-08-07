import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

import skimage.io as io
import skimage.transform as trans

from math import sqrt
import sys
import os
from random import randint

def draw_and_calculate(path_data, final_draw):

	ifColor = 0
	cnum = len(final_draw[0].shape)
	if cnum > 3: ifColor =1

	def dist(a, b):
	  return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

	colors = []
	for i in range(30):
	  colors.append(i)
	pigment = {i: (randint(0, 255), randint(0, 255), randint(0, 255)) for i in colors}

	output_res = []
	draw_res = []
	for i in range(len(path_data)):
	  
	  temp = {} # each temp for one framw
	  draw = final_draw[i].copy()

	  if ifColor:
	  	draw_rgb = draw
	  else:
	  	draw_rgb = cv2.cvtColor(draw, cv2.COLOR_GRAY2BGR)
	  
	  for n, path in path_data[i].items():
	    pts = [np.array(path, dtype=np.int32)]
	    draw_rgb = cv2.polylines(draw_rgb, pts, isClosed=False, color=pigment[n%30], thickness=2)  # color=pigment[n]
	    draw_rgb = cv2.putText(draw_rgb, str(n), path[-1], 1, 3, (255, 255, 255), 3)

	    if len(path)>1:
	      speed = dist(path[-2], path[-1])
	      total = 0
	      for j in range(len(path)-1):
	        total += dist(path[j],path[j+1])
	   
	    else:
	      speed = 0
	      total = 0

	    net = dist(path[0], path[-1]) 
	    
	    if net: 
	      ratio = round(total/net,3)
	    else: 
	      ratio = 0

	    temp[n] = [path[-1], round(speed,3), round(total,3), round(net,3), ratio]
	    # temp = {cellId: current_center, speed, total dist, net dist, ratio}

	  #draw_rgb = cv2.putText(draw_rgb, "Frame: "+str(i), \
	           # (int(512-175),int(512-20)), 1, 2, (255, 255, 255), 3)
	  
	  output_res.append(temp)
	  draw_res.append(draw_rgb)

	  #cv2.imwrite("/content/drive/Colab/t" + str(1000+i)[1:] + "_res.jpg", draw_rgb)

	return draw_res, output_res