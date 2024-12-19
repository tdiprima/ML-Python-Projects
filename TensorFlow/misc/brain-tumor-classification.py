"""
Builds and trains a convolutional neural network model to classify MRI images into four categories of brain tumors
('glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'), and tests the model on a sample image.

TODO:
1. `mkdir -p data/brain-tumor-classification-mri`
2. Download data from https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/download?datasetVersionNumber=2
FYI - Code: https://www.kaggle.com/code/shivamagarwal29/brain-tumor
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# for dirname, dirnames, filenames in os.walk('data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


X_train = []
Y_train = []

image_size = 150

labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

for i in labels:
    folderPath = os.path.join('data/brain-tumor-classification-mri/Training', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

for i in labels:
    folderPath = os.path.join('data/brain-tumor-classification-mri/Testing', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

# Shuffle
# `sklearn.utils.shuffle`
# The `random_state` parameter allows you to provide this random seed to sklearn methods.
# To reproduce the randomness for development and testing.

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train, Y_train = shuffle(X_train, Y_train, random_state=101)
print("X_train.shape", X_train.shape)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=101)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# to_categorical
# `tf.keras.utils.to_categorical`
# Converts a class vector (integers) to binary class matrix.

y_train_new = []

for i in y_train:
    y_train_new.append(labels.index(i))

y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []

for i in y_test:
    y_test_new.append(labels.index(i))

y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

# Convolutional Neural Network

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

# Model summary
# `tf.keras.model`
print(model.summary())

# Model Compile
# `tf.keras.model`
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

"""
Model Fit `tf.keras.model`
Trains the model for a fixed number of epochs (dataset iterations).
validation_split
Float between 0 and 1. Fraction of the training data to be used as validation data.
The model will set apart this fraction of the training data, will not train on it,
and will evaluate the loss and any model metrics on this data at the end of each epoch.
"""

# Training model for 20 epochs will take forever, FYI.
num_epochs = 5
# num_epochs = 20
history = model.fit(X_train, y_train, epochs=num_epochs, validation_split=0.1)

model.save('models/brain_class_1.pth')
# model.save('models/braintumor.h5')

import matplotlib.pyplot as plt

# Plot training accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()

# Plot loss curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, loss, 'r', label="Training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.legend(loc='upper left')
plt.show()

# Show image
img = cv2.imread('data/brain-tumor-classification-mri/Training/pituitary_tumor/p (107).jpg')
img = cv2.resize(img, (150, 150))
img_array = np.array(img)
print("img_array.shape", img_array.shape)

img_array = img_array.reshape(1, 150, 150, 3)
print("img_array.shape", img_array.shape)

# Interpolation
# `interpolation='nearest'` simply displays an image without trying to interpolate between pixels if the display resolution is not the same as the image resolution.

from tensorflow.keras.preprocessing import image

img = image.load_img('data/brain-tumor-classification-mri/Training/pituitary_tumor/p (107).jpg')
plt.imshow(img, interpolation='nearest')
plt.show()

# Prediction
a = model.predict(img_array)
indices = a.argmax()
print(f"\nFor image 'p (107).jpg', model predicted: {labels[indices]}")
print(f"\nPredictions: {a}")
