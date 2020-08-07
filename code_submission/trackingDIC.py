import numpy as np
import cv2
import matplotlib.pyplot as plt

from random import randint
from math import sqrt

def cell_track(res):

	class Cell():
	    def __init__(self, param):
	        self.frame = param[0]
	        self.per_frame_Id = param[1]
	        self.linking_Id = -1
	        self.center = param[2]

	        self.width = param[3][0]
	        self.height = param[3][1]
	        self.angle = param[4]
	        self.area = self.find_area()
	        self.status = self.find_status(param[5])  

	    def update_id(self, idx):
	        self.linking_Id = idx

	    def find_area(self):
	        return self.width * self.height

	    def in_mitosis(self, circles):
	        for c in circles:
	            if cv2.pointPolygonTest(c, self.center, False) == 1:     # +1, 0, -1
	                return True
	        else:
	            return False

	    def find_status(self, circles):
	        if self.in_mitosis(circles):
	            if self.area > 4000:
	                return 1
	            else:
	                return 2
	        return 0

	def dist(a, b):
	    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


	# res[i]['draw_img'] = draw  # an image array
	# res[i]['cell_num'] = len(boxs)  # number of cells
	# res[i]['boxs'] = boxs  # list of (top_left_corner_x, y , width, height)
	# res[i]['fitRect'] = fitRect  # list of cell parameter

	maxD = 80
	cellSpaceTime = []  # [[c1, c2, ...], [c1, c2, ...], [...], ...]

	result = [res[i]['fitRect'] for i in range(len(res))]

	# initialize link_pool
	link_pool = [i for i in range(res[0]['cell_num'])]

	# initialize list
	num_of_frames = len(res)
	# the first frame
	initial_cells = []
	for idx in link_pool:
	    cell = Cell(res[0]['fitRect'][idx])
	    cell.update_id(idx)
	    initial_cells.append(cell)
	cellSpaceTime.append(initial_cells)
	# the next frame
	for frame in range(1, num_of_frames):
	    cells = [Cell(e) for e in res[frame]['fitRect']]
	    cellSpaceTime.append(cells)

	# cell tracking 
	for frame in range(num_of_frames - 1):

	    for cur in cellSpaceTime[frame]:

	        nex_0 = cellSpaceTime[frame + 1][0]
	        d_min, match = dist(cur.center, nex_0.center), nex_0

	        for nex in cellSpaceTime[frame + 1]:
	            d = dist(cur.center, nex.center)
	            if d < d_min:
	                d_min, match = d, nex

	        if d_min < maxD and not (cur.status == 1 and match.area < 4000):
	            # select the cell with the minimum distance and not divided from current cell
	            if cur.linking_Id == -1:
	                cur.update_id(link_pool[-1] + 1)
	                link_pool.append(cur.linking_Id)
	            match.update_id(cur.linking_Id)

	trajectories = {}
	for time in cellSpaceTime:
	    for cell in time:
	        if cell.linking_Id == -1:
	            continue
	        if cell.linking_Id not in trajectories:
	            trajectories[cell.linking_Id] = [(cell.frame, cell.center)]
	        else:
	            trajectories[cell.linking_Id].append((cell.frame, cell.center))

	# draw trajectories

	path_data = []

	for frameNo in range(0,len(res)):
	    #print('frame:', frameNo)
	    paths = {}
	    for idx in trajectories:
	        for i, item in enumerate(trajectories[idx]):
	            if item[0] == frameNo:
	                path = list(zip(*trajectories[idx][:i + 1]))[1]
	                paths[idx] = path
	                break

	    colors = link_pool
	    # print('num of colors:', len(colors))
	    pigment = {i: (randint(0, 255), randint(0, 255), randint(0, 255))
	               for i in colors}

	    path_data.append(paths)


	return path_data

#np.save(save_path + 's1_path.npy',path_data)
#print("saved!")
#print(len(path_data))
#print(path_data[0])




