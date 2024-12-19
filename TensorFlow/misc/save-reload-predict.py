"""
Loads MNIST data, loads a pre-trained neural network model, predicts classes for test data, and calculates
and prints the F1 score, classification report, and confusion matrix for the predictions.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load data into TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define the model architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])

# Train the model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10)

# Save the entire model to a file, in a binary format.
# model.save('my_model.h5')

# Load the trained model back in
# This function returns a new model object that has the same architecture and weights as the saved model.
model = keras.models.load_model('models/my_model.h5')

# Use the trained model to make predictions
# predictions = model.predict(x_test)
# You need to convert it like this:
predictions = np.argmax(model.predict(x_test), axis=-1)  # FIXED error below.

# Classification metrics can't handle a mix of multiclass and continuous-multioutput targets
from sklearn.metrics import f1_score, confusion_matrix, classification_report

f1 = f1_score(y_test, predictions, average='macro')
print("F1 score:", f1)
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print("\nConfusion matrix:\n", cm)
