import cv2
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split

##***** Please change your mask and raw image path here ******##
mask_path = '/content/drive/Colab/data/DIC-C2DH-HeLa/' # the DIC-C2DH-HeLa folder
raw_path = '/content/drive/Colab/data/DIC-C2DH-HeLa/' # the DIC-C2DH-HeLa folder not the sequence
# ------------------------------------------------------------ #


# DIC meta-data information
img_shape = (512, 512, 1)   # both width and height should be divisible by 16
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = img_shape
non_digit = 'abcdefghijklmnopqrstuvwxyz./_'

# train file path
#_train_01 = 'DIC-C2DH-HeLa/01'
#_train_02 = 'DIC-C2DH-HeLa/02'
_train_01 = mask_path + '01'
_train_02 = mask_path + '02'

#_seg_01_GT = 'DIC-C2DH-HeLa/01_GT/SEG'
#_seg_02_GT = 'DIC-C2DH-HeLa/02_GT/SEG'
#_seg_01_ST = 'DIC-C2DH-HeLa/01_ST/SEG'
#_seg_02_ST = 'DIC-C2DH-HeLa/02_ST/SEG'
_seg_01_GT = mask_path + '01_GT/SEG'
_seg_02_GT = mask_path + '02_GT/SEG'
_seg_01_ST = mask_path + '01_ST/SEG'
_seg_02_ST = mask_path + '02_ST/SEG'


#_track_01_GT = 'DIC-C2DH-HeLa/01_GT/TRA'
#_track_02_GT = 'DIC-C2DH-HeLa/02_GT/TRA'
_track_01_GT = mask_path + '01_new_mask'
_track_02_GT = mask_path + '02_new_mask'

# test file path
#_test_01 = 'DIC-C2DH-HeLa-2/01'
#_test_02 = 'DIC-C2DH-HeLa-2/02'

_test_01 = raw_path + 'Sequence 1'
_test_02 = raw_path + 'Sequence 2'
_test_03 = raw_path + 'Sequence 3'
_test_04 = raw_path + 'Sequence 4'

# training data
train_01 = io.ImageCollection(_train_01 + '/t*.tif')
train_02 = io.ImageCollection(_train_02 + '/t*.tif')

#######################################################################################

# define data_loading functions

def get_seg_data(reference):
    if reference == 'GT':
        seg_01 = io.ImageCollection(_seg_01_GT + '/man_seg*.tif')
        seg_02 = io.ImageCollection(_seg_02_GT + '/man_seg*.tif')
        seg1_idx = [int(f.replace(_seg_01_GT, '').strip(non_digit)) for f in seg_01.files]
        seg2_idx = [int(f.replace(_seg_02_GT, '').strip(non_digit)) for f in seg_02.files]

        X_train = np.asarray([train_01[i] for i in seg1_idx] + [train_02[i] for i in seg2_idx])
        Y_train = np.asarray(list(seg_01) + list(seg_02))

    if reference == 'ST':
        seg_01 = io.ImageCollection(_seg_01_ST + '/man_seg*.tif')
        seg_02 = io.ImageCollection(_seg_02_ST + '/man_seg*.tif')
        X_train = np.asarray(list(train_01) + list(train_02))
        Y_train = np.asarray(list(seg_01) + list(seg_02))

    return X_train, Y_train


def get_track_data():
    # track_01 = io.ImageCollection(_track_01_GT + '/man_track*.tif')
    # track_02 = io.ImageCollection(_track_02_GT + '/man_track*.tif')
    track_01 = io.ImageCollection(_track_01_GT + '/*_marker.tif')
    track_02 = io.ImageCollection(_track_02_GT + '/*_marker.tif')
    X_train = np.asarray(list(train_01) + list(train_02))
    Y_train = np.asarray(list(track_01) + list(track_02))
    return X_train, Y_train


def get_test_data():
    test_01 = io.ImageCollection(_test_01 + '/t*.tif')
    test_02 = io.ImageCollection(_test_02 + '/t*.tif')
    test_03 = io.ImageCollection(_test_03 + '/t*.tif')
    test_04 = io.ImageCollection(_test_04 + '/t*.tif')
    return np.asarray(list(test_01) + list(test_02) + list(test_03) + list(test_04))



#######################################################################################

# define some utility functions

def normalize(data):
    '''Map to -0.5 ~ 0.5'''
    return np.asarray([cv2.equalizeHist(d) / 255 - 0.5 for d in data])


def to_4D(data):
    '''Reshape to 4D tensor for training'''
    return data.reshape(data.shape + (1, ))


def binarize(data):
    '''binarize reference data, i.e. 0 or 1'''
    return np.asarray([np.where(d > 0, 1, 0) for d in data])


def get_train_val(data='seg', reference='GT', split=0, seed=1):
    if data == 'seg':
        x, y = get_seg_data(reference)
        X, Y = to_4D(normalize(x)), to_4D(binarize(y))
    if data == 'track':
        x, y = get_track_data()
        X, Y = to_4D(normalize(x)), to_4D(binarize(y))
    if data == 'test':
        return to_4D(normalize(get_test_data()))
    if split == 0:
        return X, Y
    else:
        return train_test_split(X, Y, test_size=split, random_state=seed)


