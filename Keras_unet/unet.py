import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_unet(img_shape, last_layer):
	'''This version uses UpSampling2D in stead of Conv2DTranspose for up-samling'''
    inputs = layers.Input(img_shape)

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

    if last_layer == 'sigmoid':
        outputs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c)
    if last_layer == 'softmax':
        outputs = layers.Conv2D(2, kernel_size=(1, 1), activation='softmax')(c)

    model = keras.Model(inputs, outputs)
    return model
