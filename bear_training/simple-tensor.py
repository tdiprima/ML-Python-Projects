"""
Here's a complete TensorFlow code example that loads data from MNIST,
creates a model, makes predictions, and calculates the F1-score and
prints the Confusion matrix.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# todo: save the model to a file
# model.save('model_T.h5')

# Make predictions on the test set
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='macro')
print("F1 score:", f1)

# Assuming y_test is the true labels and y_pred is the predicted labels
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion matrix:\n", cm)
