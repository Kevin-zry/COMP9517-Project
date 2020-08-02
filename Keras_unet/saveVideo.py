import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.transform as trans

image_path = '/content/drive/Colab/data/DIC-C2DH-HeLa_raw/Sequence 1'
image_data = np.asarray(list(io.ImageCollection(image_path + '/t*.tif')))

video_path = '/content/drive/Colab/video.avi'
video_fps = 10
video_size = (512,512)
video_type = cv2.VideoWriter_fourcc(*'XVID')

writer = cv2.VideoWriter(video_path, video_type, video_fps, video_size, isColor = False)
for i in range(image_data.shape[0]) :
    writer.write(image_data[i])

writer.release()
print('saved')