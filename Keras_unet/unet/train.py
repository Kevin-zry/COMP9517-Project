from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data import get_train_val
from unet import build_unet

# meta-data info
img_shape = (512, 512, 1)

# Train & validation split, input shape = (batch, height, width, channels) [Full annotation]
X_train, X_val, Y_train, Y_val = get_train_val(data='seg', reference='GT', split=0.2, seed=1)

# ------------------------------------------------------------ #
# Step 1: Randomized Data Augmentation (Rescale, Rotate, Flip) #
# ------------------------------------------------------------ #

def get_data_generator(X, Y, batch_size=8, seed=5):
	'''produce generator with data augmentation on the fly'''
	datagen_args = dict(rotation_range=360,
		            zoom_range=0.4,
		            horizontal_flip=True,
		            vertical_flip=True,
		            fill_mode='reflect')
	
	image_datagen = ImageDataGenerator(**datagen_args)
	label_datagen = ImageDataGenerator(**datagen_args)

	image_datagen.fit(X, augment=True, seed=seed)
	label_datagen.fit(Y, augment=True, seed=seed)
	image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
	label_generator = label_datagen.flow(Y, batch_size=batch_size, seed=seed)

	# combine generators into one which yields image and labels
	return zip(image_generator, label_generator)

# ----------------------------------- #
# Step 2: Compile & Train U-Net Model #
# ----------------------------------- #

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = build_unet(img_shape, last_layer='sigmoid')
model.summary()

# Compile the model
# metrics = keras.metrics.SparseTopKCategoricalAccuracy(k=1)
model.compile(optimizer='adam',
              loss='binary_crossentropy', metrics='binary_accuracy')

# Train the model (optional callbacks)
log_csv = keras.callbacks.CSVLogger('toy4_logs_with_aug.csv', separator=',')

# Fit the model with original data
# history = model.fit(X_train, Y_train, batch_size=1, epochs=30, shuffle=True,
# 					callbacks=[log_csv], validation_data=(X_val, Y_val))

# Fit the model with data augmentation
train_generator = get_data_generator(X_train, Y_train, batch_size=8, seed=1)
val_generator = get_data_generator(X_val, Y_val, batch_size=8, seed=2)

# validation_data = val_generator or (X_val, Y_val)
history = model.fit(train_generator, epochs=20, steps_per_epoch=200, shuffle=True,
					callbacks=[log_csv], validation_data=val_generator)

# Save the model
model.save('DIC_seg_model_toy_a4.h5')






