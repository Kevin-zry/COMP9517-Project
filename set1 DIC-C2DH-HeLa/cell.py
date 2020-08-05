import numpy as np
import cv2
import matplotlib.pyplot as plt

# cell parameters = (frame, per_frame_Id, center, (w, h), angle, circles)


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
        self.status = self.find_status(param[5])  # 普通状态: 0, 分裂前: 1, 分裂后: 2

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

##############################################################################


from random import randint
from math import sqrt


def dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


res = np.load('D:/Github/comp9517_project/set1 DIC-C2DH-HeLa/res/s1_draw_res_new.npy', allow_pickle=True)

# res[i]['draw_img'] = draw  # an image array
# res[i]['cell_num'] = len(boxs)  # number of cells
# res[i]['boxs'] = boxs  # list of (top_left_corner_x, y , width, height)
# res[i]['fitRect'] = fitRect  # list of cell parameter

maxD = 60
cellSpaceTime = []  # [[c1, c2, ...], [c1, c2, ...], [...], ...]

result = [res[i]['fitRect'] for i in range(84)]

# 初始化link_pool
link_pool = [i for i in range(res[0]['cell_num'])]

# 初始化细胞时空list
num_of_frames = 84
# 第一帧
initial_cells = []
for idx in link_pool:
    cell = Cell(res[0]['fitRect'][idx])
    cell.update_id(idx)
    initial_cells.append(cell)
cellSpaceTime.append(initial_cells)
# 后续帧
for frame in range(1, num_of_frames):
    cells = [Cell(e) for e in res[frame]['fitRect']]
    cellSpaceTime.append(cells)

# 细胞匹配
for frame in range(num_of_frames - 1):

    for cur in cellSpaceTime[frame]:

        nex_0 = cellSpaceTime[frame + 1][0]
        d_min, match = dist(cur.center, nex_0.center), nex_0

        for nex in cellSpaceTime[frame + 1]:
            d = dist(cur.center, nex.center)
            if d < d_min:
                d_min, match = d, nex

        if d_min < maxD and not (cur.status == 1 and match.status == 2):
            # 选择距离最小且非分裂的细胞
            if cur.linking_Id == -1:
                cur.update_id(link_pool[-1] + 1)
                link_pool.append(cur.linking_Id)
            match.update_id(cur.linking_Id)

'''
# 单独处理一下最后一帧?

'''

trajectories = {}
for time in cellSpaceTime:
    for cell in time:
        if cell.linking_Id == -1:
            continue
        if cell.linking_Id not in trajectories:
            trajectories[cell.linking_Id] = [(cell.frame, cell.center)]
        else:
            trajectories[cell.linking_Id].append((cell.frame, cell.center))

# 对于任意帧画轨迹图
#frameNo = randint(0, 83)
# frameNo = 83

path_data = []

for frameNo in range(0,84):
    print('frame:', frameNo)
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

np.save('D:/Github/comp9517_project/set1 DIC-C2DH-HeLa/res/s1_path.npy',path_data)
print("saved!")
print(len(path_data))
print(path_data[0])


    # print(len(cellSpaceTime[0]))
    # print(trajectories)
    # print([(c.per_frame_Id, c.linking_Id) for c in time for time in cellSpaceTime])

'''
draw = res[frameNo]['draw_img']
for n, path in paths.items():
    print('n:', n)
    print('path:', path)
    pts = [np.array(path, dtype=np.int32)]
    draw = cv2.polylines(draw, pts, isClosed=False,
                         color=(255, 0, 0), thickness=3)  # color=pigment[n]

plt.imshow(draw,'gray')
plt.show()
'''


'''
print(len(cellSpaceTime[1]))
print(len(cellSpaceTime))
#print(cellSpaceTime[9][0])
print(trajectories)

print(cellSpaceTime[1][0].center)
'''

#print(paths)