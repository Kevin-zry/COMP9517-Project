import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from data import get_train_val

# --------------------------------------- #
# Step 1: Load Model and Make Predictions #
# --------------------------------------- #

images, labels = get_train_val(split=0) # train
test_images = get_train_val(data='test')

# plt.imshow(np.squeeze(labels)[0], 'gray')
# plt.show()

model_fname = 'DIC_seg_model_toy_a4.h5'
model = keras.models.load_model(model_fname)

# probs = np.squeeze(model.predict(images[:10]))  # (n, 512, 512, 1)
probs = np.squeeze(model.predict(test_images[:10]))

# print(probs.shape)
print(np.amin(probs)) # lowest prob
print(np.amax(probs)) # highest prob

# below is for tracking
# for i, prob in enumerate(probs):
#     # print(f'center {i+1}', prob[256, 256])
#     predict = np.argmax(prob, axis=2)
#     predictions.append(predict)

# predictions = [np.where(prob > 0.34, 1, 0) for prob in probs]

# plt.imshow(predictions[0], 'gray')
plt.imshow(probs[0], 'gray')
plt.show()
