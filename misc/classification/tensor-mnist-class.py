"""
Trains a neural network model on the MNIST dataset to classify handwritten digits, evaluates its performance
using F1 score, classification report, and confusion matrix, and prints the results.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report

"""
Load the MNIST dataset
x = images of handwritten digits
y = the corresponding labels that indicate which digit each image represents
Train = training; test = evaluating.
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

"""
Preprocess the data by dividing the pixel values by 255 to normalize them.
The pixel values in the MNIST dataset are gray scale values between 0 and 255.
Dividing by 255 scales the values to a range between 0 and 1.
"""
x_train = x_train / 255.0
x_test = x_test / 255.0

"""
Define the model architecture
The model consists of a single Flatten layer, a Dense layer with 128 neurons
and ReLU activation, and a Dense layer with 10 neurons and softmax activation.
Softmax normalizes the output of the layer to a probability distribution over the classes.
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Relu outputs the input if it is positive, and 0 otherwise.
    tf.keras.layers.Dense(128, activation='relu'),
    # Last layer in the neural network:
    tf.keras.layers.Dense(10, activation='softmax')
])

"""
Compile the model, specifying the optimizer as "adam", the loss function as
sparse_categorical_crossentropy, and the metrics as "accuracy".
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the training data
model.fit(x_train, y_train, epochs=5)

# TODO
# model.save('my_model.h5')

# Use the trained model to make predictions on the test set
y_pred = model.predict(x_test)

"""
Convert the predicted probabilities to class labels
The output of the last layer in the neural network (softmax) is a probability
distribution over the classes. Taking the argmax gives the predicted class label.
"""
y_pred = np.argmax(y_pred, axis=1)

"""
Calculate F1 score
The F1-score is a metric that combines precision and recall to evaluate classification performance.
Set the average parameter to 'macro' to calculate the macro-average F1 score.
A "good" F1-score is generally the higher the better.
"""
f1 = f1_score(y_test, y_pred, average='macro')
print("F1 score:", f1)

# Pass the labels to the classification report
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", cm)
