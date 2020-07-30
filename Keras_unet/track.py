# DIC images - cell tracking with data augmentation (trained with 01/02 GT-TRA)

import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import io

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

# define some constants
img_shape = (512, 512, 1)   # both m, n divisible by 16
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = img_shape

# train file path
_train_01 = 'DIC-C2DH-HeLa/01'
_train_02 = 'DIC-C2DH-HeLa/02'

_track_01_GT = 'DIC-C2DH-HeLa/01_GT/TRA'
_track_02_GT = 'DIC-C2DH-HeLa/02_GT/TRA'

# test file path
_test_01 = 'DIC-C2DH-HeLa-2/01'
_test_02 = 'DIC-C2DH-HeLa-2/02'

# ----------------------------- #
# Step 1: Load and Prepare Data #
# ----------------------------- #

train_01 = io.ImageCollection(_train_01 + '/t*.tif')
train_02 = io.ImageCollection(_train_02 + '/t*.tif')

track_01 = io.ImageCollection(_track_01_GT + '/man_track*.tif')
track_02 = io.ImageCollection(_track_02_GT + '/man_track*.tif')

test_01 = io.ImageCollection(_test_01 + '/t*.tif')
test_02 = io.ImageCollection(_test_02 + '/t*.tif')

track_data = np.asarray([img for img in train_01] + [img for img in train_02])
track_ref = np.asarray([img for img in track_01] + [img for img in track_02])

# --------------------------------------------------- #
# Step 2: Data Normalization - Histogram Equalization #
# --------------------------------------------------- #

# map to -0.5 ~ 0.5
track_data = np.asarray(
    [cv2.equalizeHist(img) / 255 - 0.5 for img in track_data])

# ------------------------ #
# Step 3: Reference Output #
# ------------------------ #

marker_seq = []

for sample in track_ref:
    x = list(sample[sample > 0])
    marker_seq.append(set(x))

cell_class = marker_seq[0].union(*marker_seq[1:])

# print(cell_class)
# for marker in marker_seq[:20]:
#     print(marker)

# ------------------------------------------------------------ #
# Step 4: Randomized Data Augmentation (Rescale, Rotate, Flip) #
# ------------------------------------------------------------ #

# Reshape to 4D tensor for training, input shape = (batch, height, width, channels)
track_data = track_data.reshape(track_data.shape + (1, ))
track_ref = track_ref.reshape(track_ref.shape + (1, ))

# Train & validation split
data_train, data_val, ref_train, ref_val = train_test_split(
    track_data, track_ref, test_size=14, random_state=1)

# Data augmentation on the fly
datagen_args = dict(rotation_range=360,
                    zoom_range=0.4,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect')
image_datagen = ImageDataGenerator(**datagen_args)
marker_datagen = ImageDataGenerator(**datagen_args)

seed = 1
image_datagen.fit(data_train, augment=True, seed=seed)
marker_datagen.fit(ref_train, augment=True, seed=seed)
image_generator = image_datagen.flow(data_train, batch_size=8, seed=seed)
marker_generator = marker_datagen.flow(ref_train, batch_size=8, seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, marker_generator)

# ---------------------------- #
# Step 5: U-Net Initialization #
# ---------------------------- #

def build_unet(img_shape, num_classes):
    inputs = layers.Input(img_shape)
    # s = layers.Lambda(lambda x: x / 255)(inputs)

    ### [First half of the network: downsampling inputs] ###

    # First convolution layer
    c = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      kernel_initializer='he_normal', padding='same')(inputs)
    c = layers.Dropout(0.1)(c)
    c = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      kernel_initializer='he_normal', padding='same')(c)

    convnet = {'inputs': inputs, 'c0': c}

    # MaxPooling + convolution for different feature depth
    for i, filters in enumerate([64, 128, 256, 512], start=1):
        p = layers.MaxPooling2D(pool_size=2, padding='valid')(c)
        c = layers.Conv2D(filters, kernel_size=(
            3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p)
        if i == 1:
            c = layers.Dropout(0.1)(c)
        elif i <= 3:
            c = layers.Dropout(0.2)(c)
        else:
            c = layers.Dropout(0.3)(c)
        c = layers.Conv2D(filters, kernel_size=(
            3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)

        convnet[f'c{i}'] = c
    ### [Second half of the network: upsampling inputs] ###

    # Upsampling + concatenate + convolution
    for i, filters in enumerate([256, 128, 64, 32], start=1):
        u = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c)
        t = layers.concatenate([u, convnet[f'c{4-i}']])
        c = layers.Conv2D(filters, kernel_size=(
            3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
        c = layers.Dropout(0.2)(c) if i <= 2 else layers.Dropout(0.1)(c)
        c = layers.Conv2D(filters, kernel_size=(
            3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)

    outputs = layers.Conv2D(num_classes, kernel_size=(
        1, 1), activation='softmax')(c)

    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = build_unet(img_shape, max(cell_class) + 1)
model.summary()

# --------------------- #
# Step 6: Loss Function #
# --------------------- #

# define custom loss function 
# but for now keras 'sparse_categorical_crossentropy' is used

def pixel_weight(q):
    pass


def weighted_cross_entropy(p, y):
    '''returns an array of losses (one of sample in the input batch)'''
    pass

# --------------------------------- #
# Step 7: Training (Adam Optimizer) #
# --------------------------------- #

metrics = keras.metrics.SparseTopKCategoricalAccuracy(k=1)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=metrics)

# Add callbacks
# path to save the model file
checkpoint_filepath = 'saved_models/checkpoints-{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoints = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True)
log_csv = keras.callbacks.CSVLogger('my_logs.csv', separator=',')

callbacks = [checkpoints, log_csv]

# Train the model (total of 12,800 iterations), doing validation at the end of each epoch.
history = model.fit(train_generator, verbose=1, epochs=32, steps_per_epoch=400,
                    validation_data=(data_val, ref_val), callbacks=callbacks)

model.save('DIC_tracking_model_v0.h5')

# ---------------------------------- #
# Step 8: Predict & Model Evaluation #
# ---------------------------------- #

print(history.history)

# To make predictions, use model.predict()
# To evaluate model, use model.evaluate()
# look into my_logs.csv to see how metrics varies w.r.t epoches (plot this)

# To load a saved model, use m = keras.models.load_model(filename)

