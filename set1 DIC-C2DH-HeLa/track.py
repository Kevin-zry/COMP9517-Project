import cv2
# from data import get_seg_data
import numpy as np
import matplotlib.pyplot as plt

from visualize import show_video, show_video_compare
from dc3 import draw_contours

import sys


def load_image_sequence(path, ext, size, flag=None):
    seq = []
    for i in range(size):
        if flag == 'color':
            img = cv2.imread(path + f'{i:0>2d}' + ext)
        else:
            img = cv2.imread(path + f'{i:0>2d}' + ext, -1)
        seq.append(img)
    return seq


dirpath = 'DIC-C2DH-HeLa/01/t0'
# raw images
img_seq = load_image_sequence(dirpath, '.tif', 84)
# predict images
pre_seq = load_image_sequence('mask_probs/mask_prob_', '.png', 84)
# predict markers
m_seq = load_image_sequence('m1/t0', '_marker.tif', 84)
# raw images(color)
img_color_seq = load_image_sequence(dirpath, '.tif', 84, flag='color')


# print(m_seq)
# sys.exit(0)

# result: list of (draw, ellipse_list)
result = []
box_draws = []

# Create a black image
black = np.zeros((512, 512, 3), np.uint8)

for i in range(84):
    img, img_color = img_seq[i], img_color_seq[i]
    pre, m = pre_seq[i], m_seq[i]
    draw, ellipse_list = draw_contours(img, pre, m, img_color)
    box_draws.append(draw)
    result.append(ellipse_list)


initial_cells = next(zip(*result[0]))  # identified by their elliptical centers
space_time = []  # (id, frame, (x, y))

# smallest displacement method for cell association

from math import sqrt


def dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def not_close_to_boundary(point, th=25):
    low, high = th, 512 - th
    # print('point:', point)
    if point[0] < low or point[0] > high:
        return False
    if point[1] < low or point[1] > high:
        return False
    return True


def mitosis_signal(point, black_circle):
    return cv2.pointPolygonTest(black_circle, point, False)  # +1, 0, -1


def before_mitosis(point, black_circle):
    pass


# show_video(box_draws)
# sys.exit(0)

draw = black
# draws = []
for i in range(10):
    cur_coordinates = next(zip(*result[i]))
    next_coordinates = next(zip(*result[i + 1]))
    # filter out those points close to boundary
    # print('cur_coordinates:', cur_coordinates)
    cur_to_match = list(filter(not_close_to_boundary, cur_coordinates))
    next_to_match = list(filter(not_close_to_boundary, next_coordinates))
    for cur in cur_to_match:
        d_min, closest = dist(cur, next_to_match[0]), next_to_match[0]
        # print('d_min:', d_min, '\nclosest:', closest)
        for nex in next_to_match:
            d = dist(cur, nex)
            if d < d_min:
                # print('d:', d, '\nnex:', nex)
                d_min, closest = d, nex
        cur_int = (round(cur[0]), round(cur[1]))
        closest_int = (round(closest[0]), round(closest[1]))
        # cv2.line(img[i], cur_int, closest_int, (255, 0, 0), 3)
        # draw = cv2.line(draw, cur_int, closest_int, (0, 0, 255), 2)
        print('d_min:', d_min, 'cur:', cur_int, 'closest:', closest_int)
        # draws.append(draw)
        # trajectories

# plt.imshow(draw, 'gray')
# plt.show()
