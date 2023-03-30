# TODO: Save or load, take your pick.
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define and train a model
model = keras.Sequential([
    # keras.layers.Dense(10, input_shape=(2,)),  # found shape=(32, 28, 28)
    keras.layers.Dense(10, input_shape=(28, 28)),
    keras.layers.Dense(1)
])

# model.compile(optimizer='adam', loss='mse')
# model.fit(x_train, y_train, epochs=10)

# Save the model to a file
# model.save('models/my_model.h5')

# Load the saved model from a file
loaded_model = keras.models.load_model('models/my_model.h5')

# Make predictions with your model
predictions = loaded_model.predict(x_test)

# Print out some results
print('\nPredictions:', predictions[:5])

import numpy as np

# GOTTA DO THIS BEFORE F1 SCORE, CLASSIFICATION REPORT, AND CONFUSION MATRIX
predictions = np.argmax(predictions, axis=1)

f1 = f1_score(y_test, predictions, average='macro')
print("\nF1 score:", f1)

# Pass the labels to the classification report
print("\n", classification_report(y_test, predictions))

# Calculate confusion matrix
conf_mat = confusion_matrix(y_test, predictions)

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('\nConfusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Define class labels
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Get the distinct label names
class_names = np.unique(y_train)

# Convert them to strings
class_names = class_names.astype(str)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
